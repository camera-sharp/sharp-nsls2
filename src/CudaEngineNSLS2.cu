
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

//

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


// 

void CudaEngineNSLS2::cal_object_trans(const DeviceRange<cusp::complex<float> > & input_frames,
				bool global_sync){
  if(global_sync){
    frameOverlapNormalize(input_frames.data(), m_global_image_scale.data(), m_image.data());
    m_comm->allSum(m_image);
  }else{
    frameOverlapNormalize(input_frames.data(), m_image_scale.data(), m_image.data());
  }

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

  bool newMethod = false;

  if(newMethod) {
    updateIllumination(m_image, frames_data);
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
    // std::cout << "regulization: " << regularization << std::endl;

    m_comm->allSum(illumination_numerator);
    m_comm->allSum(illumination_denominator);

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

  calculateImageScale();
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






