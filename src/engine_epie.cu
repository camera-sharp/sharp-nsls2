
#include <engine_epie.h>
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

CudaEngineEPIE::CudaEngineEPIE()
 : CudaEngine(){ 
}

//

void CudaEngineEPIE::iterate(int steps){

  m_del_image.resize(m_image.size());
  m_del_illumination.resize(m_illumination.size());

  std::cout << "CudaEngineEPIE::iterate" << std::endl;
  std::cout << "image, size: "  << m_image.shape(0) << ", " << m_image.shape(1) << std::endl;
  std::cout << "frames, size: " << m_frames.size() << std::endl;

  m_data_tolerance     = Options::getOptions()->data_tolerance;
  m_overlap_tolerance  = Options::getOptions()->overlap_tolerance;
  m_solution_tolerance = Options::getOptions()->solution_tolerance;

  int success = 0;

  // Large GPU arrays, with size equal to number of frames 
 
  thrust::device_vector<cusp::complex<float> > tmp1(m_frames.size());
  thrust::device_vector<cusp::complex<float> > tmp2(m_frames.size());
  thrust::device_vector<cusp::complex<float> > tmp3(m_frames.size());

  clock_t start_timer = clock(); //Start 

  for(int i = 0; i < steps;i++){

    bool calculate_residual = (m_iteration % m_output_period == 0);

    // Make sure to do a global synchronization in the last iteration
    bool do_sync = ((m_iteration % m_global_period) == 0 ||  i == steps-1);

    start_timer();

    float data_residual = 0;
    if(calculate_residual){
      calcDataProjector(m_frames_iterate, tmp2, &data_residual);
    }else{
      calcDataProjector(m_frames_iterate, tmp2);
    }

    thrust::transform(tmp2.begin(),
		      tmp2.end(),
		      m_frames_iterate.begin(),
		      m_frames_iterate.begin(),
		      thrust::minus<cusp::complex<float> >());

    // Calculate m_del_image and m_del_illumination

    calcDelImage(m_frames_iterate, m_del_image, do_sync);
    calcDelProbe(m_frames_iterate, tmp1, tmp2, tmp3);

    // Update image and probe

    add(m_illumination, m_del_illumination, m_illumination);
    calculateImageScale();
    add(m_image, m_del_image, m_image);

    // OverlapProjector

    float overlap_residual = 0;

    imageSplit(m_image, m_frames_iterate);
    illuminationMultiply(m_frames_iterate, m_frames_iterate);

    // print residuals

    if(calculate_residual){
         success = printDiagmostics(data_residual, overlap_residual);
    }

   if(m_iteration) {
     // if(success) { 
     //	  break;
     // }
   }

    m_iteration++;      
  }    

  if(m_comm->isLeader()){
    printSummary(success);
  }

  double diff = (clock() - start_timer) / (double) CLOCKS_PER_SEC; 
  Counter::getCounter()->addCount("total ", diff*1000);
}


// 

void CudaEngineEPIE::calcDelProbe(const DeviceRange<cusp::complex<float> > & frames_data, 
     const DeviceRange<cusp::complex<float> > & frames_object,
     const DeviceRange<cusp::complex<float> > & frames_numerator,
     const DeviceRange<cusp::complex<float> > & frames_denominator) {

  bool newMethod = false;

  if(newMethod) {
    // updateIllumination(m_image, frames_data);
  } else {

    imageSplit(m_image, frames_object);

    {
      // frames_numerator = frames_data *conj(frames_object)
      conjugateMultiply(frames_data, frames_object, frames_numerator);

      // frames_denominator = frames_object *= conj(frames_object)
      conjugateMultiply(frames_object, frames_object, frames_denominator);
    }

    thrust::device_vector<cusp::complex<float> > illumination_numerator(m_illumination.size());
    shiftedSum(frames_numerator, illumination_numerator);

    thrust::device_vector<cusp::complex<float> > illumination_denominator(m_illumination.size());
    shiftedSum(frames_denominator, illumination_denominator);

    cusp::complex<float> regularization = thrust::reduce(illumination_denominator.begin(),
							 illumination_denominator.end(),
							 cusp::complex<float> (0,0),
							 cusp::blas::detail::maximum<cusp::complex<float> >());
    regularization = 1e-4f*regularization;

    // 

    m_comm->allSum(illumination_numerator);
    m_comm->allSum(illumination_denominator);

    // thrust::device_vector<cusp::complex<float> > del_illumination(m_illumination.size());

    thrust::transform(illumination_numerator.begin(),
		      illumination_numerator.end(),
		      illumination_denominator.begin(),
		      m_del_illumination.begin(),
		      DivideWithRegularization<cusp::complex<float> >(regularization));
  }
}

void CudaEngineEPIE::calcDelImage(const DeviceRange<cusp::complex<float> > & input_frames,
     				  const DeviceRange<cusp::complex<float> > & output_image,
				  bool global_sync){

  if(global_sync){
    frameOverlapNormalize(input_frames.data(), m_global_image_scale.data(), output_image.data());
    m_comm->allSum(output_image);
  }else{
    frameOverlapNormalize(input_frames.data(), m_image_scale.data(), output_image.data());
  }

}


//

int  CudaEngineEPIE::printDiagmostics(float data_residual, float overlap_residual){

     int success = 0;

      if(m_comm->isLeader()){

	if(m_has_solution){

	  double image_residual = 0; // compareImageSolution();

	  if(m_io) { 
	    m_io->printResiduals(m_iteration, data_residual, overlap_residual, image_residual); 
	  } else {
	    sharp_log("iter = %d, data = %e, overlap = %e solution = %e (nmse)",m_iteration,data_residual, overlap_residual,image_residual);
	  }
	  if(image_residual < m_solution_tolerance){
	    success = 1;
	  }

	} else {
	  if(m_io) { 
	    m_io->printResiduals(m_iteration,data_residual,overlap_residual);
	  } else {
	    sharp_log("iter = %d, data = %e, overlap = %e",m_iteration,data_residual, overlap_residual);
	  }
	}
      
	if(data_residual < m_data_tolerance){
	  success = 1;
	}

	if(overlap_residual < m_overlap_tolerance){
	  success = 1;
	}

      }

      m_comm->allSum(success);
      return success;
}



void CudaEngineEPIE::printSummary(int success){

    if(success){
      sharp_log("Success in %d iterations...residuals beat (%e,%e,%e)", 
		m_iteration, m_data_tolerance, m_overlap_tolerance, m_solution_tolerance);
    }else{
      sharp_log("Maximum iteration exceeded. residual tolerances not reached (%e,%e,%e)", 
		m_data_tolerance, m_overlap_tolerance, m_solution_tolerance);
    }
}

void CudaEngineEPIE::calcDataProjector(const DeviceRange<cusp::complex<float> > & input_frames,
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


 if(output_residual){

    // if(!(m_background_retrieval)){
    //   cusp::blas::detail::scal(output_frames.begin(), output_frames.end(),
    //			       1.0f/sqrtf(m_frame_width*m_frame_height));
    // }

    if(!m_background_retrieval){
      // This calculates norm((output_frames - input_frames)^2)/norm(m_frames); without using additional variables     
      *output_residual = sqrt(thrust::transform_reduce(zip2(output_frames, input_frames),
						    Subtract2<thrust::tuple<cusp::complex<float>, 
						                            cusp::complex<float> > >(), 
						    float(0),
						    thrust::plus<float>()))/m_frames_norm;
    }else{
      std::cout << "TBD" << std::endl;
    }
  }



  double diff = ( clock() - start_timer ) / (double)CLOCKS_PER_SEC; 
  Counter::getCounter()->addCount("data_projector", diff*1000); 
}


