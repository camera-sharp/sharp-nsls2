
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

//

void CudaEngineNSLS2::iterate(int steps){

  std::cout << "CudaEngineNSLS2::iterate" << std::endl;

  m_data_tolerance     = Options::getOptions()->data_tolerance;
  m_overlap_tolerance  = Options::getOptions()->overlap_tolerance;
  m_solution_tolerance = Options::getOptions()->solution_tolerance;

  int success = 0;

  // Large GPU arrays, with size equal to number of frames 
 
  thrust::device_vector<cusp::complex<float> > overlap_proj(m_frames_iterate.begin(), m_frames_iterate.end());
  thrust::device_vector<cusp::complex<float> > data_proj(m_frames_iterate.begin(), m_frames_iterate.end());
  thrust::device_vector<cusp::complex<float> > composite_proj(m_frames.size());

  clock_t start_timer = clock(); //Start 

  for(int i = 0; i < steps;i++){

    bool calculate_residual = (m_iteration % m_output_period == 0);

    // Make sure to do a global synchronization in the last iteration
    bool do_sync = ((m_iteration % m_global_period) == 0 ||  i == steps-1);

    start_timer();

    // Update the image and probe

    calcImage(m_frames_iterate, m_image, do_sync);

    if(m_iteration && m_illumination_refinement_period && 
       (m_iteration % m_illumination_refinement_period) == 0){
      sharp_log("Refining illumination");
      refineIllumination(m_frames_iterate,m_image);
    } 

    // Calculate the overlap projection

    float overlap_residual = 0;
    if(calculate_residual){
      calcOverlapProjection(m_frames_iterate, m_image, overlap_proj, &overlap_residual);
    }else{
      calcOverlapProjection(m_frames_iterate, m_image, overlap_proj);
    }

   // Calculate the data projection

    float data_residual = 0;
    if(calculate_residual){
      dataProjector(m_frames_iterate, data_proj, &data_residual);
    }else{
      dataProjector(m_frames_iterate, data_proj);
    }

    dataProjector(overlap_proj, composite_proj);
    cusp::blas::scal(composite_proj, 2.0f);

    // z(i+1) = z(i) + 2.0*composite_proj - data_proj - overlap_proj

    thrust::transform(m_frames_iterate.begin(),
		      m_frames_iterate.end(),
		      composite_proj.begin(),
		      m_frames_iterate.begin(),
		      thrust::plus<cusp::complex<float> >());
    thrust::transform(m_frames_iterate.begin(),
		      m_frames_iterate.end(),
		      data_proj.begin(),
		      m_frames_iterate.begin(),
		      thrust::minus<cusp::complex<float> >());
    thrust::transform(m_frames_iterate.begin(),
		      m_frames_iterate.end(),
		      overlap_proj.begin(),
		      m_frames_iterate.begin(),
		      thrust::minus<cusp::complex<float> >());

    // print residuals

    if(calculate_residual){
	success = printDiagmostics(data_residual, overlap_residual);
    }

   if(m_iteration) {
     if(success) { 
	  break;
     }
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

void CudaEngineNSLS2::calcImage(const DeviceRange<cusp::complex<float> > & input_frames,
				const DeviceRange<cusp::complex<float> > & output_image,
				bool global_sync){
  if(global_sync){
    frameOverlapNormalize(input_frames.data(), m_global_image_scale.data(), output_image.data());
    m_comm->allSum(output_image);
  }else{
    frameOverlapNormalize(input_frames.data(), m_image_scale.data(), output_image.data());
  }

}

void CudaEngineNSLS2::calcOverlapProjection(const DeviceRange<cusp::complex<float> > & input_frames,
     				    	    const DeviceRange<cusp::complex<float> > & input_image,
				    	    const DeviceRange<cusp::complex<float> > & output_frames,
				    	    float * output_residual){
  clock_t start_timer = clock();

  if(!output_residual){
    imageSplitIlluminate(input_image.data(),output_frames.data());
  }else{
    thrust::device_vector< float > temp(1.);
    imageSplitIlluminateCompare(input_image.data(), input_frames.data(), output_frames.data(),thrust::raw_pointer_cast(&temp[0]));
    start_timer();
    thrust::host_vector< float > temp1 = temp;
    float jk = temp1[0];
    jk=sqrt(jk)/m_frames_norm;
    output_residual[0] = jk;    
    stop_timer("residual_overlap");
  }

  double diff = ( clock() - start_timer ) / (double)CLOCKS_PER_SEC; 
  Counter::getCounter()->addCount("overlap_projector", diff*1000); 
}

//

int  CudaEngineNSLS2::printDiagmostics(float data_residual, float overlap_residual){

     int success = 0;

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
	  thrust::tuple<cusp::complex<double>,double> result  = 
                 thrust::transform_reduce(zip3(m_solution, m_illuminated_area, m_image),
                 ComputeImageResidual< cusp::complex<float>, cusp::complex<double>, double >(),
		 thrust::tuple<cusp::complex<double>,double>(0),
		 TupleAdd< thrust::tuple<cusp::complex<double>, double> >());

	  double image_residual = (m_image_norm_2* thrust::get<1>(result) - 
                 (cusp::norm(thrust::get<0>(result)))) / (m_image_norm_2 * thrust::get<1>(result));

	  if (image_residual < 0) {  // numerical precision issues, violating Cauchy-Schwartz inequality
	    image_residual *= -1;
	  }
	  image_residual = sqrtf(image_residual);

	  if(m_io) { 
	    m_io->printResiduals(m_iteration, data_residual, overlap_residual, image_residual); 
	  } else {
	    sharp_log("iter = %d, data = %e, overlap = %e solution = %e (nmse)",m_iteration,data_residual, overlap_residual,image_residual);
	  }
	  if(image_residual < m_solution_tolerance){
	    success = 1;
	  }
	}else{
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

void CudaEngineNSLS2::printSummary(int success){

    if(success){
      sharp_log("Success in %d iterations...residuals beat (%e,%e,%e)", 
		m_iteration, m_data_tolerance, m_overlap_tolerance, m_solution_tolerance);
    }else{
      sharp_log("Maximum iteration exceeded. residual tolerances not reached (%e,%e,%e)", 
		m_data_tolerance, m_overlap_tolerance, m_solution_tolerance);
    }
}



