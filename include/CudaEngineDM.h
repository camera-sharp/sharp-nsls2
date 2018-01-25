#pragma once

#ifndef CUDA_ENGINE_DM_H
#define CUDA_ENGINE_DM_H

#include "cuda_engine.h"

class CudaEngineDM: public CudaEngine
{
 public:

  /** Constructor */
  CudaEngineDM();

 public:

  // Engine API
  
  /** recon: recon_ptycho */ 
  virtual void iterate(int steps);

 public:

  // New fine-grained API

  /** allocates containers */
  void init();

  /** recon: recon_dm_trans -> recon_dm_trans_single */
  int step();

 public:

   // Recon Input API

  /** general feedback parameter */
  void setBeta(float v);

  /** espresso threshold coefficient */
  void setAlpha(float v);

  /** iteration number start updating probe */
  void setStartUpdateProbe(int v);

  /** iteration number start updating object */
  void setStartUpdateObject(int v);

  /** maximum object magnitude */
  void setAmpMax(float v);

  /** minimum object magnitude */
  void setAmpMin(float v);

  /** maximum object phase */
  void setPhaMax(float v);

  /** minimum object phase */
  void setPhaMin(float v);

  /** initial probe */
  void setInitProbe(const boost::multi_array<std::complex<float>, 2> & probe);

  /** initial object */
  void setInitObject(const boost::multi_array<std::complex<float>, 2> & object); 

 public:

  // Recon Output API

  boost::multi_array<std::complex<float>, 2> & getObject();

  boost::multi_array<std::complex<float>, 2> & getProbe();

  float getObjectError() const;

  float getProbeError() const;

 public:

  // SHARP internal containers

  boost::multi_array<std::complex<float>, 3> & getFrames();

  boost::multi_array<std::complex<float>, 1> & getFramesCorners();

  // used in cal_object_trans

  boost::multi_array< int, 1> & getOverlapingFrames();

  boost::multi_array< int, 1> & getOverlapingFramesIndex();
  
  boost::multi_array<std::complex<float>, 2> & getIlluminatedArea();
  
  boost::multi_array<std::complex<float>, 2> & getImageScale();

  // used in cal_probe_trans

  boost::multi_array<std::complex<float>, 2> & getIlluminationNumerator();

  boost::multi_array<std::complex<float>, 2> & getIlluminationDenominator();

  // tmp containers

  boost::multi_array<std::complex<float>, 3> & getPrbObj();
  boost::multi_array<std::complex<float>, 3> & getTmp2();
  
 public:

  /** number of chunks */
  void setChunks(int nparts); 

 protected:

  // recon_ptycho methods

  void recon_dm_trans_single();

  void cal_object_trans(const DeviceRange<cusp::complex<float> > & input_frames,
		   bool global_sync);
  
  void set_object_constraints();

  void cal_probe_trans( 
		   const DeviceRange<cusp::complex<float> > & frames_object,
		   const DeviceRange<cusp::complex<float> > & frames_numerator,
		   const DeviceRange<cusp::complex<float> > & frames_denominator);

  void cal_obj_error(const DeviceRange<cusp::complex<float> > & obj_old);

  void cal_prb_error(const DeviceRange<cusp::complex<float> > & prb_old);

  double cal_chi_error(const DeviceRange<cusp::complex<float> > & input_image, 
		       const DeviceRange<cusp::complex<float> > & tmp_frames);

 protected:

  // local versions of sharp methods/wrappers

  void dataProjector(const DeviceRange<cusp::complex<float> > & input_frames,
		     const DeviceRange<cusp::complex<float> > & output_frames,
		     int iframe);

  void calcOverlapProjection(const DeviceRange<cusp::complex<float> > & input_image,
			     const DeviceRange<cusp::complex<float> > & output_frames,
			     int ipart);

  void calculateImageScale();

 protected:

  // debugging 

  double cal_sol_error();

 public:

  // input recon_ptycho parameters

  int m_start_update_probe;  // iteration number start updating probe, 2
  int m_start_update_object; // iteration number start updating object, 0

  float m_alpha; // espresso threshold coefficient, 1.e-8
  float m_beta;  // general feedback parameter, 0.9

  float m_amp_max;  //  maximum object magnitude, 1.0
  float m_amp_min;  //  minimum object magnitude, 0.0
  float m_pha_max;  //  maximum object phase, pi/2
  float m_pha_min;  // minimum object phase, -pi/2

  float m_sigma1;  // normalization weighting factor, 1.e-10
  float m_sigma2;  // normalization weighting factor, 5.e-5

  boost::multi_array<cusp::complex<float>, 2> m_init_probe;
  boost::multi_array<cusp::complex<float>, 2> m_init_object;

 public:

  // output

  float m_obj_error;
  float m_prb_error; 

 public:

  // scan points
  
  boost::multi_array<std::complex<float>, 1> m_host_corners;

  // containers used in cal_object_trans

  boost::multi_array<int, 1> m_host_overlaping_frames;
  boost::multi_array<int, 1> m_host_overlaping_frames_index;  

  boost::multi_array<std::complex<float>, 2> m_host_image_scale;
  boost::multi_array<std::complex<float>, 2> m_host_illuminated_area;

  // containers used in cal_probe_trans

  boost::multi_array<std::complex<float>, 2> m_host_illumination_numerator;
  boost::multi_array<std::complex<float>, 2> m_host_illumination_denominator;

  // tmp containers

  boost::multi_array<std::complex<float>, 3> m_host_prb_obj;
  boost::multi_array<std::complex<float>, 3> m_host_tmp2;

 protected:

    DeviceRange<cusp::complex<float> > m_image_old; 
    DeviceRange<cusp::complex<float> > m_prb_old;
    
    // DeviceRange<cusp::complex<float> > m_prb_tmp; 

    // Large GPU arrays, with size equal to number of frames 
 
    // thrust::device_vector<cusp::complex<float> > m_prb_obj;
    // thrust::device_vector<cusp::complex<float> > m_tmp;
    // thrust::device_vector<cusp::complex<float> > m_tmp2;

    // divided by parts

    int m_nparts;

    thrust::device_vector<cusp::complex<float> > m_prb_obj_part;
    thrust::device_vector<cusp::complex<float> > m_tmp_part;
    thrust::device_vector<cusp::complex<float> > m_tmp2_part;

    thrust::device_vector<cusp::complex<float> > m_illumination_numerator;
    thrust::device_vector<cusp::complex<float> > m_illumination_denominator;

};

#endif
