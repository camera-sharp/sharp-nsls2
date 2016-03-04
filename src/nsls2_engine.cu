
#include <nsls2_engine.h>
#include <boost/assert.hpp>
#include <cusp/blas.h>
#include <cusp/print.h>
#include <cusp/array2d.h>
#include <cmath>
#include <sstream>
#include <string>
#include "communicator.h"
#include "input_output.h"
#include "counter.h"
#include "logging.h"
#include <float.h>
#include <limits>
#include "geometry.h"

using namespace camera_sharp;

// typedef cusp::array1d<cusp::complex<float>,cusp::host_memory> cusp_complex_array;

CudaEngineNSLS2::CudaEngineNSLS2()
 : CudaEngine(){ 
}

void CudaEngineNSLS2::iterate(int steps){

  std::cout << "CudaEngineNSLS2::iterate" << std::endl;

  float data_tolerance = Options::getOptions()->data_tolerance;
  float overlap_tolerance = Options::getOptions()->overlap_tolerance;
  float solution_tolerance = Options::getOptions()->solution_tolerance;
  int success = 0;

  // Large GPU arrays, with size equal to number of frames  
  thrust::device_vector<cusp::complex<float> > frames_object(m_frames_iterate.begin(), m_frames_iterate.end());
  thrust::device_vector<cusp::complex<float> > frames_data(m_frames.size());
  //frames_object = m_frames_iterate;

  clock_t start_timer = clock();//Start 


  for(int i = 0; i<steps;i++){
    bool calculate_residual = (m_iteration % m_output_period == 0);
    float data_residual=0;

    // Make sure to do a global synchronization in the last iteration
    bool do_sync = ((m_iteration % m_global_period) == 0 ||  i == steps-1);

    if(calculate_residual){
      dataProjector(m_frames_iterate,frames_data,&data_residual);
    }else{
      dataProjector(m_frames_iterate,frames_data);
    }

    start_timer();
    // frames_iterate = (frames_iterate-frames_object)*m_beta+frames_data*(1-2*m_beta)
    cusp::blas::axpbypcz(m_frames_iterate, frames_object, frames_data, m_frames_iterate, m_beta,-m_beta, (1-2*m_beta));    
    stop_timer("axpbypcz");

    float overlap_residual = 0;
    if(calculate_residual){
      overlapProjector(frames_data, frames_object, m_image,do_sync, &overlap_residual);
    }else{
      overlapProjector(frames_data, frames_object, m_image, do_sync);
    }

    if(m_iteration && m_illumination_refinement_period && (m_iteration % m_illumination_refinement_period) == 0){
      // note that this will overwrite the contents of frames_data
      sharp_log("Refining illumination");
      refineIllumination(frames_data,m_image);
    }      

    restart_timer();
    // the line below is equivalent to m_frames_iterate += frames_object*(2*m_beta);      
    cusp::blas::axpby(m_frames_iterate, frames_object, m_frames_iterate, 1.0f, 2.0f*m_beta);
    stop_timer("axpby");

    if(calculate_residual){
      if(m_comm->isLeader()){
	if(m_has_solution){
	  // we need to compute a weighted sum and figure out the global phase difference
	  // there could be an amplitude difference as well if the illumination power is unknown.
	  //
	  // residual = sqrt( min_phi  sum | phi m_image -  m_solution |^2 illuminated_area) / m_frames_norm
	  //
	  // we also know that  m_frames_norm= sqrt(sum illuminated_area (|m_image|^2) )
	  //
	  //
	  // min_c  sum illuminated_area (|m_solution|^2 + |phi|^2 |m_image|^2 - 2 Re  c <m_image , m_solution> )
	  // we also know that  m_frames_norm= sum illuminated_area (|m_image|^2) 
	  // and c=  <m_image , m_solution>_illuminated_area / |m_solution|^2_illuminated_area 

	  // this computes sum abs(image)^2 illuminated_area and sum ( conj(image)*solution*illuminated_area)
	  thrust::tuple<cusp::complex<double>,double> result  = thrust::transform_reduce(zip3(m_solution, m_illuminated_area, m_image),
											 ComputeImageResidual< cusp::complex<float>,cusp::complex<double>, double >(),
											 thrust::tuple<cusp::complex<double>,double>(0),
											 TupleAdd< thrust::tuple<cusp::complex<double>, double> >());
	  double image_residual = (m_image_norm_2* thrust::get<1>(result) - (cusp::norm(thrust::get<0>(result)))) / (m_image_norm_2 * thrust::get<1>(result));
	  if (image_residual < 0) {  // numerical precision issues, violating Cauchy-Schwartz inequality
	    image_residual *= -1;
	  }
	  image_residual = sqrtf(image_residual);

	  if(m_io) { 
	    m_io->printResiduals(m_iteration,data_residual,overlap_residual,image_residual); 
	  } else {
	    sharp_log("iter = %d, data = %e, overlap = %e solution = %e (nmse)",m_iteration,data_residual, overlap_residual,image_residual);
	  }
	  if(image_residual < solution_tolerance){
	    success = 1;
	  }
	}else{
	  if(m_io) { 
	    m_io->printResiduals(m_iteration,data_residual,overlap_residual);
	  } else {
	    sharp_log("iter = %d, data = %e, overlap = %e",m_iteration,data_residual, overlap_residual);
	  }
	}
      
	if(data_residual < data_tolerance){
	  success = 1;
	}
	if(overlap_residual < overlap_tolerance){
	  success = 1;
	}
      }
      m_comm->allSum(success);
      if(success) { 
	break;
      }
    }
    m_iteration++;      
  }    
  if(m_comm->isLeader()){
    if(success){
      sharp_log("Success in %d iterations...residuals beat (%e,%e,%e)", 
		m_iteration, data_tolerance, overlap_tolerance, solution_tolerance);
    }else{
      sharp_log("Maximum iteration exceeded. residual tolerances not reached (%e,%e,%e)", 
		data_tolerance, overlap_tolerance, solution_tolerance);
    }
  }
  double diff = (clock() - start_timer) / (double)CLOCKS_PER_SEC; 
  Counter::getCounter()->addCount("total ", diff*1000);
}

void ThrustEngine::dataProjector(const DeviceRange<cusp::complex<float> > & input_frames,
				 const DeviceRange<cusp::complex<float> > & output_frames,
				 float * output_residual){

  clock_t start_timer = clock();//Start 

  fftFrames(input_frames.data(),
	    output_frames.data(),
	    FFT_FORWARD);
  if(m_background_retrieval){
    cusp::blas::detail::scal(output_frames.begin(), output_frames.end(), 
			     1.0f/sqrtf(m_frame_width*m_frame_height));
  }

  if(output_residual){

    if(!(m_background_retrieval)){
      cusp::blas::detail::scal(output_frames.begin(), output_frames.end(),
			       1.0f/sqrtf(m_frame_width*m_frame_height));
    }

    if(!m_background_retrieval){
      // This calculates norm(abs(output_frames)-m_frames)/norm(m_frames); without using additional variables     
      *output_residual = sqrt(thrust::transform_reduce(zip2(output_frames, m_frames),
						    AbsSubtract2<thrust::tuple<cusp::complex<float>,float> >(), 
						    float(0),
						    thrust::plus<float>()))/m_frames_norm;
    }else{
      start_timer();
      *output_residual = 0;
      typedef thrust::device_vector<float >::iterator Iterator;
      TiledRange<Iterator> repeated_mean_bg(m_mean_bg_frames.begin(), m_mean_bg_frames.end(), m_nframes);

      *output_residual = (thrust::transform_reduce(zip3(output_frames, m_frames, repeated_mean_bg),
						AbsSubtract3<thrust::tuple<cusp::complex<float>,float, float> >(), 
						float(0),
						thrust::plus<float>()));
      *output_residual = sqrt(*output_residual)/m_frames_norm;
      stop_timer("residual_data");
    }
  }

  start_timer();
  // Apply modulus projection
  if(!m_background_retrieval){
    thrust::transform(output_frames.begin(),output_frames.end() , m_frames.begin(), 
		      output_frames.begin(), DataProj<cusp::complex<float>,float >());  
  }else{
    typedef thrust::device_vector<float >::iterator Iterator;
    TiledRange<Iterator> repeated_mean_bg(m_mean_bg_frames.begin(), m_mean_bg_frames.end(), m_nframes);
    thrust::device_vector<float> signal(m_frames.size());
    thrust::device_vector<float> fourierIntensity(m_frames.size());
    {
      thrust::device_vector<float> etai(m_frame_width*m_frame_height);
      thrust::device_vector<float> etai_num(m_frame_width*m_frame_height);
      thrust::device_vector<float> etai_denom(m_frame_width*m_frame_height);
      {
	thrust::for_each(zip5(output_frames, fourierIntensity, signal, m_frames, repeated_mean_bg),
			 FourierIntensitySignalCalculator());
	
	///G2 = fourierIntensity = output_frames*output_frames, d = signal = m_frames - repeated_mean_bg;
	etaiCalculator(thrust::raw_pointer_cast(fourierIntensity.data()),
		       thrust::raw_pointer_cast(signal.data()),
		       thrust::raw_pointer_cast(etai_num.data()),
		       thrust::raw_pointer_cast(etai_denom.data()));
	
	
	m_comm->allSum(etai_num);
	m_comm->allSum(etai_denom);
	
	///divide with cutoff..
	float cutoff = 0.9;
	//tempEta = ((isfinite(tempEta)) && (tempEta >= cutoff))? tempEta : cutoff;
	thrust::transform(etai_num.begin(), etai_num.end(),
			  etai_denom.begin(), etai.begin(),
			  DivideWithCutoff<float>(cutoff));
      }
      {
	TiledRange<thrust::device_vector<float>::iterator > repeated_etai(etai.begin(), etai.end(), m_nframes);
	thrust::device_vector<float> backgroundIncrement(m_frames.size());
	thrust::device_vector<float> shiftedbackgroundIncrement(m_mean_bg_frames.size());
	thrust::for_each(zip4(backgroundIncrement, fourierIntensity, repeated_etai, signal),
			 BackgroundIncrementCalculator());
	
	averageFrame(backgroundIncrement,shiftedbackgroundIncrement);
	m_comm->allSum(shiftedbackgroundIncrement);
	
	thrust::transform(m_mean_bg_frames.begin(), m_mean_bg_frames.end(),
			  shiftedbackgroundIncrement.begin(), m_mean_bg_frames.begin(),
			  AddNonnegativeFinite<float>());
	
	thrust::for_each(zip4(output_frames, fourierIntensity, m_frames, repeated_mean_bg),
			 ScaleFourierIntensity());
      }
    }
  }
  stop_timer("data_projector_transform");

  fftFrames(output_frames.data(),
	    output_frames.data(),
	    FFT_INVERSE);
  // Scale the result
  restart_timer();
  cusp::blas::detail::scal(output_frames.begin(), output_frames.end(), 
			   1.0f/sqrtf(m_frame_width*m_frame_height));
  stop_timer("fftscalingx2")

  double diff = ( clock() - start_timer ) / (double)CLOCKS_PER_SEC; 
  Counter::getCounter()->addCount("data_projector", diff*1000); 
}
