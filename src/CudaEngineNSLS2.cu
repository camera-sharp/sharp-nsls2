
#include <CudaEngineNSLS2.h>
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
#include <sys/time.h>

#include "sharp_thrust.h"

using namespace camera_sharp;

template <typename T>
struct SetMaxAmp : public thrust::unary_function<T, T>
{

  float amp_max;
  SetMaxAmp(float _amp_max)
    : amp_max(_amp_max){}

  __host__ __device__
  T operator()(T x){
    float abs_x = cusp::abs(x);
    if(abs_x > amp_max) return x*amp_max/abs_x;
    return x;
  }
};

template <typename T>
struct SetMinAmp : public thrust::unary_function<T, T>
{

  float amp_min;
  SetMinAmp(float _amp_min)
    : amp_min(_amp_min){}

  __host__ __device__
  T operator()(T x){
    float abs_x = cusp::abs(x);
    if(abs_x < amp_min) return x*amp_min/abs_x;
    return x;
  }
};

template <typename T>
struct SetMaxPha : public thrust::unary_function<T, T>
{

  float pha_max;
  SetMaxPha(float _pha_max)
    : pha_max(_pha_max){}

  __host__ __device__
  T operator()(T x){
    float abs_x = cusp::abs(x);
    float pha_x = cusp::arg(x);
    if(pha_x > pha_max) return cusp::polar(abs_x, pha_max);
    return x;
  }
};

template <typename T>
struct SetMinPha : public thrust::unary_function<T, T>
{

  float pha_min;
  SetMinPha(float _pha_min)
    : pha_min(_pha_min){}

  __host__ __device__
  T operator()(T x){
    float abs_x = cusp::abs(x);
    float pha_x = cusp::arg(x);
    if(pha_x < pha_min) return cusp::polar(abs_x, pha_min);
    return x;
  }
};

template <typename T>
struct AbsDiff2 : public thrust::unary_function<T,float>
{
  __host__ __device__
  float operator()(T x){
    float ret = cusp::abs(thrust::get<0>(x) - thrust::get<1>(x));
    return ret*ret;
  }
};

template <typename T>
struct Norm : public thrust::unary_function<T, float>{
  __host__ __device__ 
  float operator()(T x){
    float ret = cusp::abs(x);
    return ret*ret;
  }
};

// typedef cusp::array1d<cusp::complex<float>,cusp::host_memory> cusp_complex_array;

CudaEngineNSLS2::CudaEngineNSLS2()
 : CudaEngine(){ 

   m_start_update_object = 0;
   m_start_update_probe  = 2;

   m_alpha = 1.e-8;
   m_beta  = 0.9;

   m_amp_max =  1.0;
   m_amp_min =  0.0;
   m_pha_max =  3.14/2;
   m_pha_min = -3.14/2;
}

// Recon API

void CudaEngineNSLS2::setAlpha(float v){
     m_alpha = v;
}

void CudaEngineNSLS2::setBeta(float v){
     m_beta = v;
}

void CudaEngineNSLS2::setStartUpdateProbe(int v){
     m_start_update_probe = v;
}

void CudaEngineNSLS2::setStartUpdateObject(int v){
     m_start_update_object = v;
}

void CudaEngineNSLS2::setAmpMax(float v){
     m_amp_max = v;
}

void CudaEngineNSLS2::setAmpMin(float v){
     m_amp_min = v;
}


void CudaEngineNSLS2::setPhaMax(float v){
     m_pha_max = v;
}

void CudaEngineNSLS2::setPhaMin(float v){
     m_pha_min = v;
}


//

/*

void CudaEngineNSLS2::iterate(int steps){

  std::cout << "CudaEngineNSLS2::iterate" << std::endl;
  // std::cout << "image: "  << m_image.shape(0) << ", " << m_image.shape(1) << std::endl;
  // std::cout << "frames: " << m_frames.size() << std::endl;


  thrust::device_vector<cusp::complex<float> > image_old(m_image.size()); 
  thrust::device_vector<cusp::complex<float> > prb_old(m_illumination.size()); 
  thrust::device_vector<cusp::complex<float> > prb_tmp(m_illumination.size()); 

  // Large GPU arrays, with size equal to number of frames 
 
  thrust::device_vector<cusp::complex<float> > prb_obj(m_frames.size());
  thrust::device_vector<cusp::complex<float> > tmp(m_frames.size());
  thrust::device_vector<cusp::complex<float> > tmp2(m_frames.size());

  clock_t start_timer = clock(); //Start 

  for(int i = 0; i < steps;i++){

    // double sol_err = cal_sol_error();
    // std::cout << "sol_chi: " << sol_err << std::endl;

    start_timer();

    bool calculate_residual = (m_iteration % m_output_period == 0);

    // Make sure to do a global synchronization in the last iteration
    bool do_sync = ((m_iteration % m_global_period) == 0 ||  i == steps-1);

    //

    thrust::copy(m_image.begin(), m_image.end(), image_old.data());
    thrust::copy(m_illumination.begin(), m_illumination.end(), prb_old.data());

    // Calculate the overlap projection: 
    // prb_obj = self.prb * self.obj[x_start:x_end, y_start:y_end]

    calcOverlapProjection(m_frames_iterate, m_image, prb_obj);

    // tmp = 2. * prb_obj - self.product[i]

    thrust::copy(prb_obj.begin(), prb_obj.end(), tmp.begin());
    cusp::blas::scal(tmp, 2.0f);  

    thrust::transform(tmp.begin(),
		      tmp.end(),
		      m_frames_iterate.begin(),
		      tmp.begin(),
		      thrust::minus<cusp::complex<float> >());

   // Calculate the data projection: tmp2

   dataProjector(tmp, tmp2);

    // result = self.beta * (tmp2 - prb_obj)

    thrust::transform(tmp2.begin(),
		      tmp2.end(),
		      prb_obj.begin(),
		      tmp.begin(),
		      thrust::minus<cusp::complex<float> >());

    cusp::blas::scal(tmp, m_beta);

    // z(i+1) = z(i) + result

    thrust::transform(m_frames_iterate.begin(),
		      m_frames_iterate.end(),
		      tmp.begin(),
		      m_frames_iterate.begin(),
		      thrust::plus<cusp::complex<float> >());

    // Update m_image and m_illumination
    // prb_obj, tmp, tmp2 - temporary containers

    if(m_iteration >= m_start_update_probe) {
      if(m_iteration >= m_start_update_object) {
          cal_object_trans(m_frames_iterate, do_sync);
	  set_object_constraints();
          cal_probe_trans(m_frames_iterate, prb_obj, tmp, tmp2, prb_tmp);
      } else {
          cal_probe_trans(m_frames_iterate, prb_obj, tmp, tmp2, prb_tmp);
      }
    } else {
      if(m_iteration >= m_start_update_object){
          cal_object_trans(m_frames_iterate, do_sync);
	  set_object_constraints();
      }
    }
  
    // print residuals

    if(calculate_residual){

	double obj_err = cal_obj_error(image_old);
	double prb_err = cal_prb_error(prb_old);
	double chi_err = cal_chi_error(m_image, tmp);
 
	std::cout << m_iteration 
	          << ", object_chi: " << obj_err 
		  << ", probe_chi: " << prb_err
		  << ", diff_chi: " << chi_err << std::endl;
    }

    m_iteration++;      
  }    

  if(m_comm->isLeader()){
    // printSummary(success);
  }

  double diff = (clock() - start_timer) / (double) CLOCKS_PER_SEC; 
  Counter::getCounter()->addCount("total ", diff*1000);
}

*/

void CudaEngineNSLS2::iterate(int steps){


  clock_t start_timer = clock(); //Start 

  for(int i = 0; i < steps; i++) {
   	  step();
       	  // double diff =  (clock() - start_timer)/(double) CLOCKS_PER_SEC; 
       	  // std::cout << i << ", time: " << diff << std::endl;
       	  // start_timer = clock();
    }
}

void CudaEngineNSLS2::init(){

  std::cout << "CudaEngineNSLS2::init" << std::endl;

  // std::cout << "image: "  << m_image.shape(0) << ", " << m_image.shape(1) << std::endl;
  // std::cout << "frames: " << m_frames.size() << std::endl;

  m_image_old.resize(m_image.size()); 
  m_prb_old.resize(m_illumination.size()); 
  m_prb_tmp.resize(m_illumination.size()); 

  // Large GPU arrays, with size equal to number of frames 
 
  m_prb_obj.resize(m_frames.size());
  m_tmp.resize(m_frames.size());
  m_tmp2.resize(m_frames.size());

  // moved fftPlan from CudaEngine

  int fft_dims[2] = {m_frame_width,m_frame_height};
  int batch = m_nframes;
  cufftResult status = cufftPlanMany(&m_fftPlan, 2, fft_dims, NULL,1,0,NULL,1,0,CUFFT_C2C,batch);

  m_iteration = 0;
}

int CudaEngineNSLS2::step(){

    // double sol_err = cal_sol_error();
    // std::cout << "sol_chi: " << sol_err << std::endl;

    // start_timer();
    // double diff = 0.0;
    double start_timer = clock();

    struct timeval  start_tv;
    gettimeofday(&start_tv, NULL);

    bool calculate_residual = (m_iteration % m_output_period == 0);

    // Make sure to do a global synchronization in the last iteration
    bool do_sync = true; // ((m_iteration % m_global_period) == 0 ||  i == steps-1);

    //

    thrust::copy(m_image.begin(), m_image.end(), m_image_old.data());
    thrust::copy(m_illumination.begin(), m_illumination.end(), m_prb_old.data());

    cudaDeviceSynchronize();

    struct timeval  copy_tv;
    gettimeofday(&copy_tv, NULL);

    // printf ("SharpNSLS2::step, copy, time(tv): = %f seconds\n",
    //     (double) (copy_tv.tv_usec - start_tv.tv_usec) / 1000000 +
    //     (double) (copy_tv.tv_sec - start_tv.tv_sec));

    double copy_timer = clock();
    // diff =  (copy_timer - start_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::step, copy, time: %e \n", diff);

    // Calculate the overlap projection: 
    // prb_obj = self.prb * self.obj[x_start:x_end, y_start:y_end]

    calcOverlapProjection(m_frames_iterate, m_image, m_prb_obj);

    // cudaThreadSynchronize();
    cudaDeviceSynchronize();

    struct timeval  overlap_tv;
    gettimeofday(&overlap_tv, NULL);

    // printf ("SharpNSLS2::step, overlap, time(tv): = %f seconds\n",
    //     (double) (overlap_tv.tv_usec - copy_tv.tv_usec) / 1000000 +
    //     (double) (overlap_tv.tv_sec - copy_tv.tv_sec));

    double overlap_timer = clock();
    // diff =  (overlap_timer - copy_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::step, overlap, time: %e \n", diff);

    // tmp = 2. * prb_obj - self.product[i]

    thrust::copy(m_prb_obj.begin(), m_prb_obj.end(), m_tmp.begin());
    cusp::blas::scal(m_tmp, 2.0f);  

    thrust::transform(m_tmp.begin(),
		      m_tmp.end(),
		      m_frames_iterate.begin(),
		      m_tmp.begin(),
		      thrust::minus<cusp::complex<float> >());

    cudaDeviceSynchronize();

    struct timeval  trans_tv;
    gettimeofday(&trans_tv, NULL);

    // printf ("SharpNSLS2::step, trans, time(tv): = %f seconds\n",
    //     (double) (trans_tv.tv_usec - overlap_tv.tv_usec) / 1000000 +
    //     (double) (trans_tv.tv_sec - overlap_tv.tv_sec));

    double trans_timer = clock();
    // diff =  (trans_timer - overlap_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::step, trans, time: %e \n", diff);

    // Calculate the data projection: tmp2

    dataProjector(m_tmp, m_tmp2);

    cudaDeviceSynchronize();

    struct timeval  proj_tv;
    gettimeofday(&proj_tv, NULL);

    // printf ("SharpNSLS2::step, proj, time(tv): = %f seconds\n",
    //     (double) (proj_tv.tv_usec - trans_tv.tv_usec) / 1000000 +
    //     (double) (proj_tv.tv_sec - trans_tv.tv_sec));

    double proj_timer = clock();
    // diff =  (proj_timer - trans_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::step, proj, time: %e \n", diff);

    // result = self.beta * (tmp2 - prb_obj)

    thrust::transform(m_tmp2.begin(),
		      m_tmp2.end(),
		      m_prb_obj.begin(),
		      m_tmp.begin(),
		      thrust::minus<cusp::complex<float> >());

    cusp::blas::scal(m_tmp, m_beta);

    // z(i+1) = z(i) + result

    thrust::transform(m_frames_iterate.begin(),
		      m_frames_iterate.end(),
		      m_tmp.begin(),
		      m_frames_iterate.begin(),
		      thrust::plus<cusp::complex<float> >());

    // Update m_image and m_illumination
    // prb_obj, tmp, tmp2 - temporary containers

    cudaDeviceSynchronize();

    struct timeval  trans2_tv;
    gettimeofday(&trans2_tv, NULL);

    // printf ("SharpNSLS2::step, trans2, time(tv): = %f seconds\n",
    //     (double) (trans2_tv.tv_usec - proj_tv.tv_usec) / 1000000 +
    //     (double) (trans2_tv.tv_sec - proj_tv.tv_sec));

    // printf ("SharpNSLS2::step, local, time(tv): = %f seconds\n",
    //       (double) (trans2_tv.tv_usec - start_tv.tv_usec) / 1000000 +
    //       (double) (trans2_tv.tv_sec - start_tv.tv_sec));

    double trans2_timer = clock();
    // diff =  (trans2_timer - proj_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::step, trans2, time: %e \n", diff);

    if(m_iteration >= m_start_update_probe) {
      if(m_iteration >= m_start_update_object) {

          cal_object_trans(m_frames_iterate, do_sync);

    	  cudaDeviceSynchronize();

    	  struct timeval  object_tv;
    	  gettimeofday(&object_tv, NULL);

    	  // printf ("SharpNSLS2::step, object, time(tv): = %f seconds\n",
          // 	 (double) (object_tv.tv_usec - trans2_tv.tv_usec) / 1000000 +
          //	 (double) (object_tv.tv_sec - trans2_tv.tv_sec));

    	  double object_timer = clock();
    	  // diff =  (object_timer - trans2_timer)/(double) CLOCKS_PER_SEC; 
   	  // printf("SharpNSLS2::step, object, time: %e \n", diff);

	  set_object_constraints();

	  cudaDeviceSynchronize();

   	  struct timeval  limit_tv;
    	  gettimeofday(&limit_tv, NULL);

    	  // printf ("SharpNSLS2::step, limit, time(tv): = %f seconds\n",
          //	 (double) (limit_tv.tv_usec - object_tv.tv_usec) / 1000000 +
          //	 (double) (limit_tv.tv_sec - object_tv.tv_sec));

   	  double limit_timer = clock();
    	  // diff =  (limit_timer - object_timer)/(double) CLOCKS_PER_SEC; 
  	  // printf("SharpNSLS2::step, limits, time: %e \n", diff);

          cal_probe_trans(m_frames_iterate, m_prb_obj, m_tmp, m_tmp2, m_prb_tmp);

	  cudaDeviceSynchronize();

  	  struct timeval  probe_tv;
    	  gettimeofday(&probe_tv, NULL);

    	  // printf ("SharpNSLS2::step, probe, time(tv): = %f seconds\n",
          //	 (double) (probe_tv.tv_usec - limit_tv.tv_usec) / 1000000 +
          //	 (double) (probe_tv.tv_sec - limit_tv.tv_sec));

  	  double probe_timer = clock();
    	  // diff =  (probe_timer - limit_timer)/(double) CLOCKS_PER_SEC; 
  	  // printf("SharpNSLS2::step, probe, time: %e \n", diff);

      } else {

          cal_probe_trans(m_frames_iterate, m_prb_obj, m_tmp, m_tmp2, m_prb_tmp);

	  cudaDeviceSynchronize();

 	  struct timeval  probe_tv;
    	  gettimeofday(&probe_tv, NULL);

    	  // printf ("SharpNSLS2::step, probe, time(tv): = %f seconds\n",
          //	 (double) (probe_tv.tv_usec - trans2_tv.tv_usec) / 1000000 +
          //	 (double) (probe_tv.tv_sec - trans2_tv.tv_sec));

  	  double probe_timer = clock();
    	  // diff =  (probe_timer - trans2_timer)/(double) CLOCKS_PER_SEC; 
  	  // printf("SharpNSLS2::step, probe, time: %e \n", diff);

      }
    } else {
      if(m_iteration >= m_start_update_object){

          cal_object_trans(m_frames_iterate, do_sync);

	  cudaDeviceSynchronize();

 	  struct timeval  object_tv;
    	  gettimeofday(&object_tv, NULL);

    	  // printf ("SharpNSLS2::step, object, time(tv): = %f seconds\n",
          //	 (double) (object_tv.tv_usec - trans2_tv.tv_usec) / 1000000 +
          //	 (double) (object_tv.tv_sec - trans2_tv.tv_sec));


   	  double object_timer = clock();
    	  // diff =  (object_timer - trans2_timer)/(double) CLOCKS_PER_SEC; 
   	  // printf("SharpNSLS2::step, object, time: %e \n", diff);

	  set_object_constraints();

	  cudaDeviceSynchronize();

	  struct timeval  limit_tv;
    	  gettimeofday(&limit_tv, NULL);

    	  // printf ("SharpNSLS2::step, limit, time(tv): = %f seconds\n",
          //	 (double) (limit_tv.tv_usec - object_tv.tv_usec) / 1000000 +
          //	 (double) (limit_tv.tv_sec - object_tv.tv_sec));

  	  double limit_timer = clock();
    	  // diff =  (limit_timer - object_timer)/(double) CLOCKS_PER_SEC; 
  	  // printf("SharpNSLS2::step, limits, time: %e \n", diff);
      }
    }

   cudaDeviceSynchronize();

   struct timeval  update_tv;
   gettimeofday(&update_tv, NULL);

   double update_timer = clock();
   // diff =  (update_timer - trans2_timer)/(double) CLOCKS_PER_SEC; 
   // std::cout << "SharpNSLS2::step, update, time: " << diff << std::endl;
  
    // print residuals

    if(calculate_residual){

	double obj_err = cal_obj_error(m_image_old);
	double prb_err = cal_prb_error(m_prb_old);
	double chi_err = cal_chi_error(m_image, m_tmp);
 
	std::cout << m_iteration 
	          << ", object_chi: " << obj_err 
		  << ", probe_chi: " << prb_err
		  << ", diff_chi: " << chi_err << std::endl;
    }

   cudaDeviceSynchronize();

    // struct timeval  err_tv;
    // gettimeofday(&err_tv, NULL);

    // printf ("SharpNSLS2::step, err, time(tv): = %f seconds\n",
    //     (double) (err_tv.tv_usec - update_tv.tv_usec) / 1000000 +
    //     (double) (err_tv.tv_sec - update_tv.tv_sec));

    // printf ("SharpNSLS2::step, step, time(tv): = %f seconds\n",
    //     (double) (err_tv.tv_usec - start_tv.tv_usec) / 1000000 +
    //     (double) (err_tv.tv_sec - start_tv.tv_sec));

   // double err_timer = clock();
   // diff =  (err_timer - update_timer)/(double) CLOCKS_PER_SEC; 
   // printf("SharpNSLS2::step, err, time: %e \n", diff);

   // diff =  (err_timer - start_timer)/(double) CLOCKS_PER_SEC; 
   // printf("SharpNSLS2::step, step, time: %e \n", diff);

    m_iteration++;
    return m_iteration;
}

// 

void CudaEngineNSLS2::cal_object_trans(const DeviceRange<cusp::complex<float> > & input_frames,
				bool global_sync){

  double start_timer = clock();
  // double diff = 0.0;
  // double sum_timer = 0.0;

  if(global_sync){

    frameOverlapNormalize(input_frames.data(), m_global_image_scale.data(), m_image.data());

    double overlap_timer = clock();
    // diff =  (overlap_timer - start_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::cal_object_trans, overlap, time: %e \n", diff);

    m_comm->allSum(m_image);

    double sum_timer = clock();
    // diff =  (sum_timer - overlap_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::cal_object_trans, sum, time: %e \n", diff);

  }else{
    frameOverlapNormalize(input_frames.data(), m_image_scale.data(), m_image.data());
  }

  double end_timer = clock();
  // diff =  (end_timer - start_timer)/(double) CLOCKS_PER_SEC; 
  // printf("SharpNSLS2::cal_object_trans, total, time: %e \n", diff);

}

void CudaEngineNSLS2::set_object_constraints(){

  // ptycho_recon

   thrust::transform(m_image.begin(),m_image.end(), m_image.begin(),
		      SetMaxAmp<cusp::complex<float> >(m_amp_max));
   thrust::transform(m_image.begin(),m_image.end(), m_image.begin(),
		      SetMinAmp<cusp::complex<float> >(m_amp_min));
   thrust::transform(m_image.begin(),m_image.end(), m_image.begin(),
   		      SetMaxPha<cusp::complex<float> >(m_pha_max));
   thrust::transform(m_image.begin(),m_image.end(), m_image.begin(),
    		      SetMinPha<cusp::complex<float> >(m_pha_min));

}

void CudaEngineNSLS2::cal_probe_trans(const DeviceRange<cusp::complex<float> > & frames_data, 
     const DeviceRange<cusp::complex<float> > & frames_object,
     const DeviceRange<cusp::complex<float> > & frames_numerator,
     const DeviceRange<cusp::complex<float> > & frames_denominator,
     const DeviceRange<cusp::complex<float> > & prb_tmp) {

  double start_timer = clock();
  // double diff = 0.0;
  // double sum_timer = 0.0;

  struct timeval  start_tv;
  struct timeval  sum_tv;
  gettimeofday(&start_tv, NULL);

  bool newMethod = false;

  if(newMethod) {
    updateIllumination(m_image, frames_data);
  } else {

    imageSplit(m_image, frames_object);

    cudaDeviceSynchronize();

    struct timeval  split_tv;
    gettimeofday(&split_tv, NULL);

    // printf ("SharpNSLS2::cal_probe_trans, split, time(tv): = %f seconds\n",
    //       (double) (split_tv.tv_usec - start_tv.tv_usec) / 1000000 +
    //       (double) (split_tv.tv_sec - start_tv.tv_sec));

    double split_timer = clock();
    // diff =  (split_timer - start_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::cal_probe_trans, split, time: %e \n", diff);

    {
      // frames_numerator = frames_data *conj(frames_object)
      conjugateMultiply(frames_data, frames_object, frames_numerator);

      // frames_denominator = frames_object *= conj(frames_object)
      conjugateMultiply(frames_object, frames_object, frames_denominator);
    }

    cudaDeviceSynchronize();

    struct timeval  conj_tv;
    gettimeofday(&conj_tv, NULL);

    // printf ("SharpNSLS2::cal_probe_trans, conj, time(tv): = %f seconds\n",
    //       (double) (conj_tv.tv_usec - split_tv.tv_usec) / 1000000 +
    //       (double) (conj_tv.tv_sec - split_tv.tv_sec));

    double conj_timer = clock();
    // diff =  (conj_timer - split_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::cal_probe_trans, conj, time: %e \n", diff);

    thrust::device_vector<cusp::complex<float> > illumination_numerator(m_illumination.size());
    shiftedSum(frames_numerator, illumination_numerator);

    thrust::device_vector<cusp::complex<float> > illumination_denominator(m_illumination.size());
    shiftedSum(frames_denominator, illumination_denominator);

    cudaDeviceSynchronize();

    struct timeval  shift_tv;
    gettimeofday(&shift_tv, NULL);

    // printf ("SharpNSLS2::cal_probe_trans, shift, time(tv): = %f seconds\n",
    //       (double) (shift_tv.tv_usec - conj_tv.tv_usec) / 1000000 +
    //       (double) (shift_tv.tv_sec - conj_tv.tv_sec));

    double shift_timer = clock();
    // diff =  (shift_timer - conj_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::cal_probe_trans, shift, time: %e \n", diff);

    cusp::complex<float> regularization = thrust::reduce(illumination_denominator.begin(),
							 illumination_denominator.end(),
							 cusp::complex<float> (0,0),
							 cusp::blas::detail::maximum<cusp::complex<float> >());
    regularization = 1e-4f*regularization;
    // std::cout << "regulization: " << regularization << std::endl;

    cudaDeviceSynchronize();

    struct timeval  regul_tv;
    gettimeofday(&regul_tv, NULL);

    // printf ("SharpNSLS2::cal_probe_trans, regul, time(tv): = %f seconds\n",
    //       (double) (regul_tv.tv_usec - shift_tv.tv_usec) / 1000000 +
    //       (double) (regul_tv.tv_sec - shift_tv.tv_sec));

    double regul_timer = clock();
    // diff =  (regul_timer - shift_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::cal_probe_trans, regul, time: %e \n", diff);

    m_comm->allSum(illumination_numerator);
    m_comm->allSum(illumination_denominator);

    cudaDeviceSynchronize();

    gettimeofday(&sum_tv, NULL);

    // printf ("SharpNSLS2::cal_probe_trans, sum, time(tv): = %f seconds\n",
    //       (double) (sum_tv.tv_usec - regul_tv.tv_usec) / 1000000 +
    //       (double) (sum_tv.tv_sec - regul_tv.tv_sec));

    // sum_timer = clock();
    // diff =  (sum_timer - regul_timer)/(double) CLOCKS_PER_SEC; 
    // printf("SharpNSLS2::cal_probe_trans, sum, time: %e \n", diff);

    // ptycho

/*
    thrust::copy(m_illumination.begin(), m_illumination.end(), prb_tmp.data());
    cusp::blas::scal(prb_tmp, 0.1f);  

    thrust::transform(illumination_numerator.begin(),
		      illumination_numerator.end(),
		      prb_tmp.begin(),
		      illumination_numerator.begin(),
		      thrust::plus<cusp::complex<float> >());

    regularization = 0.1;
*/

    // 

    thrust::transform(illumination_numerator.begin(),
		      illumination_numerator.end(),
		      illumination_denominator.begin(),
		      m_illumination.begin(),
		      DivideWithRegularization<cusp::complex<float> >(regularization));
  }

  cudaDeviceSynchronize();

    struct timeval  trans_tv;
    gettimeofday(&trans_tv, NULL);

    // printf ("SharpNSLS2::cal_probe_trans, trans, time(tv): = %f seconds\n",
    //       (double) (trans_tv.tv_usec - sum_tv.tv_usec) / 1000000 +
    //       (double) (trans_tv.tv_sec - sum_tv.tv_sec));

  double trans_timer = clock();
  // diff =  (trans_timer - sum_timer)/(double) CLOCKS_PER_SEC; 
  // printf("SharpNSLS2::cal_probe_trans, trans, time: %e \n", diff);

  calculateImageScale();

  cudaDeviceSynchronize();

    struct timeval  scale_tv;
    gettimeofday(&scale_tv, NULL);

    // printf ("SharpNSLS2::cal_probe_trans, scale, time(tv): = %f seconds\n",
    //       (double) (scale_tv.tv_usec - trans_tv.tv_usec) / 1000000 +
    //       (double) (scale_tv.tv_sec - trans_tv.tv_sec));

 
    // printf ("SharpNSLS2::cal_probe_trans, total, time(tv): = %f seconds\n",
    //       (double) (scale_tv.tv_usec - start_tv.tv_usec) / 1000000 +
    //       (double) (scale_tv.tv_sec - start_tv.tv_sec));


  double scale_timer = clock();
  // diff =  (scale_timer - trans_timer)/(double) CLOCKS_PER_SEC; 
  // printf("SharpNSLS2::cal_probe_trans, scale, time: %e \n", diff);

  // diff =  (scale_timer - start_timer)/(double) CLOCKS_PER_SEC; 
  // printf("SharpNSLS2::cal_probe_trans, total, time: %e \n", diff);
}


double CudaEngineNSLS2::cal_obj_error(const DeviceRange<cusp::complex<float> > & obj_old){

  // self.error_obj[it] = np.sqrt(np.sum(np.abs(self.obj - self.obj_old)**2)) / \
  //             np.sqrt(np.sum(np.abs(self.obj)**2))

   double diff_error = sqrt(thrust::transform_reduce(zip2(m_image, obj_old),
		AbsDiff2<thrust::tuple<cusp::complex<float>, cusp::complex<float> > >(), 
		float(0),
		thrust::plus<float>()));

   double norm = sqrt(thrust::transform_reduce(m_image.begin(), m_image.end(),
		Norm< cusp::complex<float> >(), 
		float(0),
		thrust::plus<float>()));

   return diff_error/norm;
}

double CudaEngineNSLS2::cal_prb_error(const DeviceRange<cusp::complex<float> > & prb_old){

   // self.error_prb[it] = np.sqrt(np.sum(np.abs(self.prb - self.prb_old)**2)) / \
   //       np.sqrt(np.sum(np.abs(self.prb)**2))

   double diff_error = sqrt(thrust::transform_reduce(zip2(m_illumination, prb_old),
		AbsDiff2<thrust::tuple<cusp::complex<float>, cusp::complex<float> > >(), 
		float(0),
		thrust::plus<float>()));

   double norm = sqrt(thrust::transform_reduce(m_illumination.begin(), m_illumination.end(),
		Norm< cusp::complex<float> >(), 
		float(0),
		thrust::plus<float>()));

   return diff_error/norm;
}

double CudaEngineNSLS2::cal_chi_error(const DeviceRange<cusp::complex<float> > & image,
       const DeviceRange<cusp::complex<float> > & tmp){

   // chi_tmp = 0.
   // for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
   //      tmp = np.abs(fftn(self.prb*self.obj[x_start:x_end, y_start:y_end])/np.sqrt(1.*self.nx_prb*self.ny_prb))
   //      chi_tmp = chi_tmp + np.sum((tmp - self.diff_array[i])**2)/(np.sum((self.diff_array[i])**2))
   // self.error_chi[it] = np.sqrt(chi_tmp/self.num_points)

   imageSplitIlluminate(image.data(), tmp.data());  
   fftFrames(tmp.data(), tmp.data(), FFT_FORWARD);
   cusp::blas::detail::scal(tmp.begin(), tmp.end(), 1.0f/sqrtf(m_frame_width*m_frame_height));

   double err = sqrt(thrust::transform_reduce(zip2(tmp, m_frames),
				AbsSubtract2<thrust::tuple<cusp::complex<float>,float> >(), 
				float(0),
				thrust::plus<float>()))/m_frames_norm;

   return err;
}

double CudaEngineNSLS2::cal_sol_error(){

   double diff_error = sqrt(thrust::transform_reduce(zip2(m_image, m_solution),
		AbsDiff2<thrust::tuple<cusp::complex<float>, cusp::complex<float> > >(), 
		float(0),
		thrust::plus<float>()));

   double norm = sqrt(thrust::transform_reduce(m_solution.begin(), m_solution.end(),
		Norm< cusp::complex<float> >(), 
		float(0),
		thrust::plus<float>()));

   return diff_error/norm;
}

// 

void CudaEngineNSLS2::calcOverlapProjection(const DeviceRange<cusp::complex<float> > & input_frames,
     				    	    const DeviceRange<cusp::complex<float> > & input_image,
				    	    const DeviceRange<cusp::complex<float> > & output_frames,
				    	    float * output_residual){
  clock_t start_timer = clock();

  if(!output_residual){
    imageSplitIlluminate(input_image.data(), output_frames.data());
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

/*
        norm_probe_array = np.zeros((self.nx_obj, self.ny_obj)) + self.alpha # recon.alpha = 1.e-8

        prb_sqr = np.abs(self.prb) ** 2
        prb_conj = self.prb.conjugate()
        for i, (x_start, x_end, y_start, y_end) in enumerate(self.point_info):
            norm_probe_array[x_start:x_end, y_start:y_end] += prb_sqr
            obj_update[x_start:x_end, y_start:y_end] += prb_conj * self.product[i]

        obj_update /= norm_probe_array
*/

void CudaEngineNSLS2::calculateImageScale(){

  CudaEngine::calculateImageScale();

  /*

  m_illuminated_area.resize(m_image.size());
  illuminationsOverlap(thrust::raw_pointer_cast(m_illuminated_area.data()));
  float abs_max = thrust::transform_reduce(m_illumination.begin(), m_illumination.end(),
					   Abs<cusp::complex<float> >(),
					   cusp::complex<float>(0), 
					   cusp::blas::detail::maximum<cusp::complex<float> >()).real();
  m_image_scale.resize(m_image.size());
  m_sigma = abs_max*abs_max*1e-9;

  // std::cout << "sigma: " << m_sigma << std::endl;
  m_sigma = m_alpha;

  // compute m_image_scale=1/m_illuminated_area
  thrust::transform(m_illuminated_area.begin(), 
                    m_illuminated_area.end(),
		    m_image_scale.begin(),
		    InvSigma<cusp::complex<float> >(m_sigma));

  // compute m_global_image_scale=1/all sum (m_illuminated_area)
  cusp::array1d<cusp::complex<float>,cusp::device_memory> global_m_illuminated_area(m_illuminated_area);
  m_comm->allSum(global_m_illuminated_area);
  m_global_image_scale.resize(m_image.size());
  thrust::transform(global_m_illuminated_area.begin(),
                    global_m_illuminated_area.end(),
		    m_global_image_scale.begin(),
		    InvSigma<cusp::complex<float> >(m_sigma));
*/
}






