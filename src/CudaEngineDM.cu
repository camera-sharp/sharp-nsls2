#include <CudaEngineDM.h>
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


//

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
    float abs_x = cusp::abs(x) + 1.e-8;
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

template <typename T>
struct Real : public thrust::unary_function<T, float>{
  __host__ __device__ 
  float operator()(T x){
    return x.real();
  }
};

template <typename T1,typename T2>
struct NormDiffRecon : public thrust::binary_function<T1,T2,T1>
{

  NormDiffRecon() {}

  __host__ __device__
  T1 operator()(T1 x, T2 y){
      T1 val = cusp::abs(x) - sqrtf(y);
      return val*val;
  }
};


template <typename T1,typename T2>
struct DataProjRecon : public thrust::binary_function<T1,T2,T1>
{

  T2 sigma;
  DataProjRecon(T2 _sigma)
    :sigma(_sigma){}

  __host__ __device__
  T1 operator()(T1 x, T2 y){
    if(x != T1(0)){
      return (x/(cusp::abs(x) + sigma)*sqrtf(y));
    }
    return T1(0);
  }
};

template <typename T1,typename T2>
struct DataProj3 : public thrust::binary_function<T1,T2,T1>
{
  __host__ __device__
  T1 operator()(T1 x, T2 y){
    return cusp::polar(sqrt(y), arg(x));
  }
};

/*! Functor that takes the inverse of a vector with regularization */
template <typename T>
struct InvSigmaRecon : public thrust::unary_function<T,T>
{
  T sigma;
  InvSigmaRecon(T _sigma)
    :sigma(_sigma){}

  __host__ __device__ T operator()(T x){
    return T(1)/(x+sigma);
  }
};

// typedef cusp::array1d<cusp::complex<float>,cusp::host_memory> cusp_complex_array;

CudaEngineDM::CudaEngineDM()
 : CudaEngine(){

  // Recon input parameters

   m_start_update_object = 0;
   m_start_update_probe  = 2;

   m_alpha = 1.e-8; // regularization used by claculateImageScale in InvSigmaRecon
   m_beta  = 0.9;   // general feedback parameter

   m_amp_max =  1.0;
   m_amp_min =  0.0;
   m_pha_max =  3.14/2;
   m_pha_min = -3.14/2;

   m_sigma1 = 1.e-10; // regularization used by dataProjector in DataProjRecon
   m_sigma2 = 5.e-5;

   // Recon output parameters

   m_obj_error = 0.0;
   m_prb_error = 0.0;
   
   // MPI/GPU parameters
   
   m_nparts = 1;
}

// Recon Input API

void CudaEngineDM::setStartUpdateProbe(int v){
     m_start_update_probe = v;
}

void CudaEngineDM::setStartUpdateObject(int v){
     m_start_update_object = v;
}

void CudaEngineDM::setAlpha(float v){
     m_alpha = v;
}

void CudaEngineDM::setBeta(float v){
     m_beta = v;
}

void CudaEngineDM::setAmpMax(float v){
     m_amp_max = v;
}

void CudaEngineDM::setAmpMin(float v){
     m_amp_min = v;
}


void CudaEngineDM::setPhaMax(float v){
     m_pha_max = v;
}

void CudaEngineDM::setPhaMin(float v){
     m_pha_min = v;
}

// SHARP-NSLS2 Input API

void CudaEngineDM::setChunks(int v){
     m_nparts = v;
}

// SHARP-based API

void CudaEngineDM::setInitProbe(const boost::multi_array<std::complex<float>, 2> & probe){

     int xdim = probe.shape()[0];
     int ydim = probe.shape()[1];
     m_init_probe.resize(boost::extents[xdim][ydim]);

     const std::complex<float>* data = probe.data();
     std::complex<float>* dest = (std::complex<float>*) m_init_probe.data();

     for(int ip = 0; ip < xdim*ydim; ip++){
       dest[ip] = data[ip];
     }

     ThrustEngine::setIllumination(HostRange<cusp::complex<float> >(m_init_probe));
}

void CudaEngineDM::setInitObject(const boost::multi_array<std::complex<float>, 2> & object){

     int xdim = object.shape()[0];
     int ydim = object.shape()[1];
     m_init_object.resize(boost::extents[xdim][ydim]);

     const std::complex<float>* data = object.data();
     std::complex<float>* dest = (std::complex<float>*) m_init_object.data();

     for(int ip = 0; ip < xdim*ydim; ip++){
       dest[ip] = data[ip];
     }

     ThrustEngine::setImage(HostRange<cusp::complex<float> >(m_init_object));
}

// Recon Output API

boost::multi_array<std::complex<float>, 2> & CudaEngineDM::getObject(){
        // update and return ThrustEngine::m_host_image
	return ThrustEngine::getImage();
}

boost::multi_array<std::complex<float>, 2> & CudaEngineDM::getProbe(){
        // update and return ThrustEngine::m_host_illumination
	return ThrustEngine::getIllumination();
}

float CudaEngineDM::getObjectError() const {
     return m_obj_error;
}

float CudaEngineDM::getProbeError() const {
     return m_prb_error;
}

float CudaEngineDM::getChiError() const {
     return m_chi_error;
}


// access to the SHARP internal containers

boost::multi_array<std::complex<float>, 3> & CudaEngineDM::getFrames(){
  return ThrustEngine::getFrames();
}

boost::multi_array<std::complex<float>, 1> & CudaEngineDM::getFramesCorners(){
  m_host_corners.resize(boost::extents[m_frames_corners.size()]);
  thrust::copy(m_frames_corners.begin(), m_frames_corners.end(),
	       (cusp::complex<float> *)m_host_corners.data());  
  return m_host_corners;
}

// used in cal_object_trans

boost::multi_array< int , 1> & CudaEngineDM::getOverlapingFrames(){
  m_host_overlaping_frames.resize(boost::extents[m_overlaping_frames.size()]);
  thrust::copy(m_overlaping_frames.begin(), m_overlaping_frames.end(),
	       (int *)m_host_overlaping_frames.data());  
  return m_host_overlaping_frames;
}

boost::multi_array< int , 1> & CudaEngineDM::getOverlapingFramesIndex(){
  m_host_overlaping_frames_index.resize(boost::extents[m_overlaping_frames_index.size()]);
  thrust::copy(m_overlaping_frames_index.begin(), m_overlaping_frames_index.end(),
	       (int *)m_host_overlaping_frames_index.data());  
  return m_host_overlaping_frames_index;
}


boost::multi_array<std::complex<float>, 2> & CudaEngineDM::getIlluminatedArea(){
  m_host_illuminated_area.resize(boost::extents[m_image_height][m_image_width]);
  thrust::copy(m_illuminated_area.begin(), m_illuminated_area.end(),
	       (cusp::complex<float> *) m_host_illuminated_area.data());  
  return m_host_illuminated_area;
}

boost::multi_array<std::complex<float>, 2> & CudaEngineDM::getImageScale(){
  m_host_image_scale.resize(boost::extents[m_image_height][m_image_width]);
  thrust::copy(m_image_scale.begin(), m_image_scale.end(),
	       (cusp::complex<float> *) m_host_image_scale.data());  
  return m_host_image_scale;
}

// used in cal_probe_trans

boost::multi_array<std::complex<float>, 2> & CudaEngineDM::getIlluminationNumerator(){
  m_host_illumination_numerator.resize(boost::extents[m_frame_height][m_frame_width]);
  thrust::copy(m_illumination_numerator.begin(), m_illumination_numerator.end(),
	       (cusp::complex<float> *) m_host_illumination_numerator.data());  
  return m_host_illumination_numerator;
}

boost::multi_array<std::complex<float>, 2> & CudaEngineDM::getIlluminationDenominator(){
  m_host_illumination_denominator.resize(boost::extents[m_frame_height][m_frame_width]);
  thrust::copy(m_illumination_denominator.begin(), m_illumination_denominator.end(),
	       (cusp::complex<float> *) m_host_illumination_denominator.data());  
  return m_host_illumination_denominator;
}

// tmp containers

boost::multi_array<std::complex<float>, 3> & CudaEngineDM::getPrbObj(){
  int batch = m_nframes/m_nparts;
  m_host_prb_obj.resize(boost::extents[batch][m_frame_height][m_frame_width]);
  thrust::copy(m_prb_obj_part.begin(), m_prb_obj_part.end(),
	       (cusp::complex<float> *)m_host_prb_obj.data());  
  return m_host_prb_obj;
}

boost::multi_array<std::complex<float>, 3> & CudaEngineDM::getTmp2(){
  int batch = m_nframes/m_nparts;
  m_host_tmp2.resize(boost::extents[batch][m_frame_height][m_frame_width]);
  thrust::copy(m_tmp2_part.begin(), m_tmp2_part.end(),
	       (cusp::complex<float> *)m_host_tmp2.data());  
  return m_host_tmp2;
}


// recon: recon_ptycho

void CudaEngineDM::iterate(int steps){

  // clock_t start_timer = clock(); 
  for(int i = 0; i < steps; i++) {
   	  step();
       	  // double diff =  (clock() - start_timer)/(double) CLOCKS_PER_SEC; 
       	  // std::cout << i << ", time: " << diff << std::endl;
       	  // start_timer = clock();
    }
}

void CudaEngineDM::init(){

  std::cout << "CudaEngineDM::init" << std::endl;

  calculateImageScale();

  // std::cout << "image, shape(0): "  << m_image.shape(0) << ", shape(1)" << m_image.shape(1) << std::endl;
  std::cout << "image, (w: "  << m_image_width << ", h:" << m_image_height << ")" << std::endl; 
  std::cout << "size of frames: " << m_frames.size() << ", number of chunks: " << m_nparts << std::endl;

  m_image_old.resize(m_image.size()); 
  m_prb_old.resize(m_illumination.size());

  m_illumination_numerator.resize(m_illumination.size());
  m_illumination_denominator.resize(m_illumination.size());
  
  // m_prb_tmp.resize(m_illumination.size());

  size_t free,total;
  cudaMemGetInfo(&free,&total);
  sharp_log("CudaEngineDM::step (1) after the SHARP initialization, GPU Memory: %zu bytes free %zu bytes total.",free,total);

  // from CudaEngine::initialize

  // Large GPU arrays, with size equal to number of frames 
 
  // m_prb_obj.resize(m_frames.size());
  // m_tmp.resize(m_frames.size());
  // m_tmp2.resize(m_frames.size());

  // divided by parts

  /// create fftPlan
  int fft_dims[2] = {m_frame_width,m_frame_height};
  
  int batch = m_nframes/m_nparts;
  cufftResult status = cufftPlanMany(&m_fftPlan, 2, fft_dims, NULL,1,0,NULL,1,0,CUFFT_C2C,batch);

  int part = m_frames.size()/m_nparts;

  m_prb_obj_part.resize(part);
  m_tmp_part.resize(part);  
  m_tmp2_part.resize(part);

  // std::cout << "part: " << part <<
  //	    ", frames: " << batch <<
  //	    ", frames corners: " << m_frames_corners.size() << std::endl;

  cudaMemGetInfo(&free,&total);
  sharp_log("CudaEngineDM::step (2) after addining the chunk-based containers, GPU Memory: %zu bytes free %zu bytes total.",free,total);

  m_iteration = 0;
}

// recon: recon_dm_trans -> recon_dm_trans_single

int CudaEngineDM::step(){

    // Make sure to do a global synchronization in the last iteration
    bool do_sync = true; // ((m_iteration % m_global_period) == 0 ||  i == steps-1);

    // m_image -> m_image_old (used only for errors)
    thrust::copy(m_image.begin(), m_image.end(), m_image_old.data());
    
    // m_illumination -> m_prb_old (used only for errors)
    thrust::copy(m_illumination.begin(), m_illumination.end(), m_prb_old.data());

    recon_dm_trans_single();

    if(m_iteration >= m_start_update_probe) {
       if(m_iteration >= m_start_update_object) {
          cal_object_trans(m_frames_iterate, do_sync);
	  set_object_constraints();	  
          cal_probe_trans(m_prb_obj_part, m_tmp_part, m_tmp2_part);

      } else {
          cal_probe_trans(m_prb_obj_part, m_tmp_part, m_tmp2_part);
      }
    } else {
      if(m_iteration >= m_start_update_object){
          cal_object_trans(m_frames_iterate, do_sync);
	  set_object_constraints();
      }
    }

    // calculate resudual

    cal_obj_error(m_image_old);
    cal_prb_error(m_prb_old);
    cal_chi_error();

    m_iteration++;
    
    return m_iteration;
}

// Recon-oriented version of the SHARP methods

void CudaEngineDM::calcOverlapProjection(
   const DeviceRange<cusp::complex<float> > & input_image,
   const DeviceRange<cusp::complex<float> > & output_frames,
   int iPart) {

    // cuda_engine.cu
    // imageSplitIlluminate(input_image.data(), output_frames.data());

    int pFrames = m_nframes/m_nparts;
    int iFrame = iPart*pFrames;

    // image*illumination
    image_split(output_frames.data(),
    		input_image.data(), 
                  thrust::raw_pointer_cast(m_frames_corners.data()) + iFrame, // add iFrame
                  m_image_width, m_image_height,
                  m_frame_width, m_frame_height,
		  pFrames,                                        // replace m_nframes with pFrames
		  m_wrap_around, 
                  thrust::raw_pointer_cast(m_illumination.data()));
}

void CudaEngineDM::dataProjector(const DeviceRange<cusp::complex<float> > & input_frames,
				 const DeviceRange<cusp::complex<float> > & output_frames,
				 int iframe){

  thrust::device_vector<float >::iterator frames_it0 = m_frames.begin();
  thrust::device_vector<float >::iterator frames_it1;
  frames_it1 = frames_it0 + iframe;

  fftFrames(input_frames.data(),  output_frames.data(), FFT_FORWARD);

  cusp::blas::detail::scal(output_frames.begin(), output_frames.end(), 
			   1.0f/sqrtf(m_frame_width*m_frame_height));

  // Apply modulus projection
  
   thrust::transform(output_frames.begin(), output_frames.end() ,
                     frames_it1,                                // replace m_frames.begin() with frames_it1
		     output_frames.begin(),
		     DataProjRecon<cusp::complex<float>, float >(m_sigma1));

   fftFrames(output_frames.data(), output_frames.data(), FFT_INVERSE);

   cusp::blas::detail::scal(output_frames.begin(), output_frames.end(), 
			   1.0f/sqrtf(m_frame_width*m_frame_height));

}


//

void CudaEngineDM::recon_dm_trans_single(){

    int part = m_frames.size()/m_nparts;

    thrust::device_vector<cusp::complex<float> >::iterator frames_it0 = m_frames_iterate.begin();
    thrust::device_vector<cusp::complex<float> >::iterator frames_it1;
    thrust::device_vector<cusp::complex<float> >::iterator frames_it2;
  
    for (int i=0; i < m_nparts; i++){

        cusp::blas::detail::scal(m_prb_obj_part.begin(), m_prb_obj_part.end(), 0.0f);
	cusp::blas::detail::scal(m_tmp_part.begin(), m_tmp_part.end(), 0.0f);
	cusp::blas::detail::scal(m_tmp2_part.begin(), m_tmp2_part.end(), 0.0f);

    	frames_it1 = frames_it0 + i*part;
   	frames_it2 = frames_it1 + part;

	// std::cout << i << ", (1) it2-it1: " << thrust::distance(frames_it1, frames_it2) << std::endl;

        // Calculate the overlap projection: 
        // recon: prb_obj = self.prb * self.obj[x_start:x_end, y_start:y_end]
    
	calcOverlapProjection(m_image, m_prb_obj_part, i);        // add i 

    	// recon: tmp = 2. * prb_obj - self.product[i]
    
	thrust::copy(m_prb_obj_part.begin(), m_prb_obj_part.end(), m_tmp_part.begin());
    	cusp::blas::scal(m_tmp_part, 2.0f);

    	thrust::transform(m_tmp_part.begin(),m_tmp_part.end(),
		          frames_it1,                          // replace begin with frames_it1
		          m_tmp_part.begin(),
		          thrust::minus<cusp::complex<float> >());
			  
	// std::cout << i << ", (2) it2-it1: " << thrust::distance(frames_it1, frames_it2) << std::endl;

       // Calculate the data projection: tmp2

       dataProjector(m_tmp_part, m_tmp2_part, i*part);

       // result = self.beta * (tmp2 - prb_obj)

       thrust::transform(m_tmp2_part.begin(),
		         m_tmp2_part.end(),
		         m_prb_obj_part.begin(),
		         m_tmp_part.begin(),
		         thrust::minus<cusp::complex<float> >());

       cusp::blas::scal(m_tmp_part, m_beta);


       // z(i+1) = z(i) + result

       thrust::transform(frames_it1,                  // replace begin() with it1
		      	 frames_it2,                  // replace end() with it2
		         m_tmp_part.begin(),
		         frames_it1,                  // replace begin() with it1
		         thrust::plus<cusp::complex<float> >());

      	// std::cout << i << ", (3) it2-it1: " << thrust::distance(frames_it1, frames_it2) << std::endl;

     }


}

void CudaEngineDM::cal_object_trans(const DeviceRange<cusp::complex<float> > & input_frames,
				bool global_sync){

  double start_timer = clock();
  // double diff = 0.0;
  // double sum_timer = 0.0;

  if(global_sync){

    frameOverlapNormalize(input_frames.data(), m_global_image_scale.data(), m_image.data());
    m_comm->allSum(m_image);

  }else{
    frameOverlapNormalize(input_frames.data(), m_image_scale.data(), m_image.data());
  }

}

void CudaEngineDM::set_object_constraints(){

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

void CudaEngineDM::cal_probe_trans(
     const DeviceRange<cusp::complex<float> > & frames_object,      // m_prb_obj
     const DeviceRange<cusp::complex<float> > & frames_numerator,
     const DeviceRange<cusp::complex<float> > & frames_denominator) {

    thrust::device_vector<cusp::complex<float> > illumination_numerator(m_illumination.size());
    cusp::blas::detail::scal(illumination_numerator.begin(), illumination_numerator.end(), 0.0f);
    
    thrust::device_vector<cusp::complex<float> > illumination_denominator(m_illumination.size());
    cusp::blas::detail::scal(illumination_denominator.begin(), illumination_denominator.end(), 0.0f);

    thrust::device_vector<cusp::complex<float> > illumination_numerator_part(m_illumination.size());
    thrust::device_vector<cusp::complex<float> > illumination_denominator_part(m_illumination.size());

    thrust::device_vector<cusp::complex<float> >::iterator frames_it0 = m_frames_iterate.begin();
    thrust::device_vector<cusp::complex<float> >::iterator frames_it1;
    thrust::device_vector<cusp::complex<float> >::iterator frames_it2;

    int part = m_frames.size()/m_nparts;
    int pFrames = m_nframes/m_nparts;
    int frame_size = m_illumination.size();
  
    for (int i=0; i < m_nparts; i++){

      cusp::blas::detail::scal(m_prb_obj_part.begin(), m_prb_obj_part.end(), 0.0f);
      cusp::blas::detail::scal(m_tmp_part.begin(), m_tmp_part.end(), 0.0f);
      cusp::blas::detail::scal(m_tmp2_part.begin(), m_tmp2_part.end(), 0.0f);
 
      frames_it1 = frames_it0 + i*part;
      frames_it2 = frames_it1 + part;

      int iFrame = i*pFrames;
 
      // imageSplit(m_image, frames_object);
	 
      image_split(frames_object.data(), m_image.data(), 
                thrust::raw_pointer_cast(m_frames_corners.data()) + iFrame, // add iFrame
                m_image_width, m_image_height,
                m_frame_width, m_frame_height,
		pFrames,                                                  // replace m_nframes with pFrames
		m_wrap_around);

      // frames_numerator = frames_data *conj(frames_object)
      // conjugateMultiply(frames_data, frames_object, frames_numerator);
      
      thrust::transform(frames_it1, frames_it2,
      		        frames_object.begin(),
			frames_numerator.begin(),
		        ConjugateMultiply<cusp::complex<float> >());

      // frames_denominator = frames_object *conj(frames_object)
      // conjugateMultiply(frames_object, frames_object, frames_denominator);
      
      thrust::transform(frames_object.begin(), frames_object.end(),
                        frames_object.begin(),
			frames_denominator.begin(),
		        ConjugateMultiply<cusp::complex<float> >());

       // sum frames

       cusp::blas::detail::scal(illumination_numerator_part.begin(), illumination_numerator_part.end(), 0.0f);
       cusp::blas::detail::scal(illumination_denominator_part.begin(), illumination_denominator_part.end(), 0.0f);    

       thrust::device_vector<cusp::complex<float> >::iterator frames_numerator_it0 = frames_numerator.begin();
       thrust::device_vector<cusp::complex<float> >::iterator frames_numerator_it1;
       thrust::device_vector<cusp::complex<float> >::iterator frames_numerator_it2;

       thrust::device_vector<cusp::complex<float> >::iterator frames_denominator_it0 = frames_denominator.begin();
       thrust::device_vector<cusp::complex<float> >::iterator frames_denominator_it1;
       thrust::device_vector<cusp::complex<float> >::iterator frames_denominator_it2;

       boost::multi_array<std::complex<float>, 2> host_illumination_denominator_part;
       host_illumination_denominator_part.resize(boost::extents[m_frame_height][m_frame_width]);

       for (int j = 0; j < pFrames; j++){

          // thrust::copy(illumination_denominator_part.begin(), illumination_denominator_part.end(),
	  //    (cusp::complex<float> *) host_illumination_denominator_part.data());
	  // std::cout << j << ", before: " << host_illumination_denominator_part[0][0] << std::endl;

          frames_numerator_it1 = frames_numerator_it0 + j*frame_size;
          frames_numerator_it2 = frames_numerator_it1 + frame_size;

          frames_denominator_it1 = frames_denominator_it0 + j*frame_size;
          frames_denominator_it2 = frames_denominator_it1 + frame_size;

       	  thrust::transform(frames_numerator_it1,
			    frames_numerator_it2,
		            illumination_numerator_part.begin(),                
		            illumination_numerator_part.begin(),
		            thrust::plus<cusp::complex<float> >());

          thrust::transform(frames_denominator_it1,
			    frames_denominator_it2,
		            illumination_denominator_part.begin(),                
		            illumination_denominator_part.begin(),
		            thrust::plus<cusp::complex<float> >());

         // thrust::copy(illumination_denominator_part.begin(), illumination_denominator_part.end(),
	 //      (cusp::complex<float> *) host_illumination_denominator_part.data());
	 //  std::cout << j << ", after: " << host_illumination_denominator_part[0][0] << std::endl;

      }

 /*
      // shiftedSum(frames_numerator, illumination_numerator);
      shifted_sum(frames_numerator.data(),
		  thrust::raw_pointer_cast(illumination_numerator_part.data()),
	          thrust::raw_pointer_cast(m_frames_corners.data()) + iFrame, // add iFrame
 	          m_frame_width, m_frame_height,
		  pFrames);  // replace m_nframes with pFrames


      // shiftedSum(frames_denominator, illumination_denominator);
      shifted_sum(frames_denominator.data(),
		  thrust::raw_pointer_cast(illumination_denominator_part.data()),
	          thrust::raw_pointer_cast(m_frames_corners.data()) + iFrame, // add iFrame
 	          m_frame_width, m_frame_height,
		  pFrames);  // replace m_nframes with pFrames

*/

      // sum chunks

         thrust::transform(illumination_numerator_part.begin(),
	              illumination_numerator_part.end(),
		      illumination_numerator.begin(),                
		      illumination_numerator.begin(),
		      thrust::plus<cusp::complex<float> >());

         thrust::transform(illumination_denominator_part.begin(),
	              illumination_denominator_part.end(),
		      illumination_denominator.begin(),                
		      illumination_denominator.begin(),
		      thrust::plus<cusp::complex<float> >());

    }

    cusp::complex<float> regularization = thrust::reduce(illumination_denominator.begin(),
							 illumination_denominator.end(),
							 cusp::complex<float> (0,0),
							 cusp::blas::detail::maximum<cusp::complex<float> >());
    // regularization = 1e-4f*regularization;
    // std::cout << "regulization: " << regularization << std::endl;
    regularization = 0.0;

    m_comm->allSum(illumination_numerator);
    m_comm->allSum(illumination_denominator);

    thrust::copy(illumination_numerator.begin(), illumination_numerator.end(), m_illumination_numerator.data());
    thrust::copy(illumination_denominator.begin(), illumination_denominator.end(), m_illumination_denominator.data());

    // update illumination

    thrust::transform(illumination_numerator.begin(),
		      illumination_numerator.end(),
		      illumination_denominator.begin(),
		      m_illumination.begin(),
		      DivideWithRegularization<cusp::complex<float> >(regularization));

   calculateImageScale();
}


void CudaEngineDM::cal_obj_error(const DeviceRange<cusp::complex<float> > & obj_old){

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

  m_obj_error = diff_error/norm;

  return;
}

void CudaEngineDM::cal_prb_error(const DeviceRange<cusp::complex<float> > & prb_old){

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

    m_prb_error = diff_error/norm;

   return;
}

void CudaEngineDM::cal_chi_error(){

    int frame_size = m_illumination.size();
    int pFrames = m_nframes/m_nparts; // frames/chunck
    int part = m_frames.size()/m_nparts; // (frames/chunck)*frame_size

    thrust::device_vector<float >::iterator frames_it0 = m_frames.begin();

    double chi_error  = 0.0;
  
    for (int i=0; i < m_nparts; i++){

        // clean tmp chunk-based containers

        cusp::blas::detail::scal(m_prb_obj_part.begin(), m_prb_obj_part.end(), 0.0f);
	cusp::blas::detail::scal(m_tmp_part.begin(), m_tmp_part.end(), 0.0f);
	cusp::blas::detail::scal(m_tmp2_part.begin(), m_tmp2_part.end(), 0.0f);

        // define iterators
	
	thrust::device_vector<float >::iterator frames_part_it0 = frames_it0 + i*part;
	thrust::device_vector<cusp::complex<float> >::iterator tmp2_it0 = m_tmp2_part.begin();

        // Calculate the overlap projection: 
        // recon: prb_obj = self.prb * self.obj[x_start:x_end, y_start:y_end]
    
	calcOverlapProjection(m_image, m_prb_obj_part, i);  // add i

	// propagate to detectors

	fftFrames(m_prb_obj_part.data().get(),  m_tmp_part.data().get(), FFT_FORWARD);

  	cusp::blas::detail::scal(m_tmp_part.begin(), m_tmp_part.end(), 
			   	1.0f/sqrtf(m_frame_width*m_frame_height));

        // (tmp - self.diff_array[i])**2
	
  	thrust::transform(m_tmp_part.begin(), m_tmp_part.end() ,
                          frames_part_it0,          
		          m_tmp2_part.begin(),
		          NormDiffRecon<cusp::complex<float>, float >());
  
       double chi_part_error = 0.0;     

       for(int j = 0; j < pFrames; j++){

           // np.sum((tmp - self.diff_array[i])**2)

           thrust::device_vector<cusp::complex<float> >::iterator tmp2_it1 = tmp2_it0 + j*frame_size;
	   thrust::device_vector<cusp::complex<float> >::iterator tmp2_it2 = tmp2_it1 + frame_size;

       	   double chi_frame_error = thrust::transform_reduce(tmp2_it1, tmp2_it2,
	                                                     Real < cusp::complex<float> >(), 
						             float(0),
					                     thrust::plus<float>());

           // np.sum((self.diff_array[i])**2)

           thrust::device_vector<float >::iterator frames_part_it1 = frames_part_it0 + j*frame_size;
	   thrust::device_vector<float >::iterator frames_part_it2 = frames_part_it1 + frame_size;

           double frame_norm = thrust::reduce(frames_part_it1, frames_part_it2, float(0), thrust::plus<float>());

           chi_part_error += chi_frame_error/frame_norm;

       }

       chi_error += chi_part_error;	

    }

    m_chi_error = sqrt(chi_error/m_nframes);

    return;
}

void CudaEngineDM::calculateImageScale(){

  // CudaEngine::calculateImageScale();

  m_illuminated_area.resize(m_image.size());
  illuminationsOverlap(thrust::raw_pointer_cast(m_illuminated_area.data()));
  float abs_max = thrust::transform_reduce(m_illumination.begin(), m_illumination.end(),
					   Abs<cusp::complex<float> >(),
					   cusp::complex<float>(0), 
					   cusp::blas::detail::maximum<cusp::complex<float> >()).real();
  m_image_scale.resize(m_image.size());
  m_sigma = m_alpha; // abs_max*abs_max*1e-9;

  // compute m_image_scale=1/m_illuminated_area
  thrust::transform(m_illuminated_area.begin(), 
                    m_illuminated_area.end(),
		    m_image_scale.begin(),
		    InvSigmaRecon<cusp::complex<float> >(m_sigma));

  // compute m_global_image_scale=1/all sum (m_illuminated_area)

  cusp::array1d<cusp::complex<float>,cusp::device_memory> global_m_illuminated_area(m_illuminated_area);
  m_comm->allSum(global_m_illuminated_area);


  m_global_image_scale.resize(m_image.size());
  thrust::transform(global_m_illuminated_area.begin(),
                    global_m_illuminated_area.end(),
		    m_global_image_scale.begin(),
		    InvSigmaRecon<cusp::complex<float> >(m_sigma));
}


