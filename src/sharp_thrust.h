#pragma once

#ifndef SHARP_THRUST_H
#define SHARP_THRUST_H

#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/random.h>
#include <cusp/blas.h>
#include <cusp/print.h>
#include <cusp/array2d.h>
#include <vector>

namespace camera_sharp { 

#define zip2(a, b) thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin())), \
    thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end()))
#define zip3(a, b, c) thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), c.begin())), \
    thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), c.end()))
#define zip4(a, b, c, d) thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), c.begin(), d.begin())), \
    thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), c.end(), d.end()))
#define zip5(a, b, c, d, e) thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), c.begin(), d.begin(), e.begin())), \
    thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), c.end(), d.end(), e.end()))

/************************* Start of all the Functors ***************************/

/*!
 * @brief compute  (|input(1)|-sqrt(input2)|)^2
 */
template <typename T>
struct AbsSubtract2 : public thrust::unary_function<T,float>
{
  __host__ __device__
  float operator()(T x){
    float ret = cusp::abs(thrust::get<0>(x))-sqrtf(thrust::get<1>(x));
    return ret*ret;
  }
};

/*!
 * @brief compute(input0-sqrt((input1-input2)>0) )^2 for  background subtracted discrepancy 
 */
template <typename T>
struct AbsSubtract3 : public thrust::unary_function<T,float>
{
  __host__ __device__
  float operator()(T x){
    float I = thrust::get<1>(x)-thrust::get<2>(x);
    if(I < 0){
      I = 0;
    }
    float ret = cusp::abs(thrust::get<0>(x))-sqrtf(I);
    return ret*ret;
  }
};

/*!
 * @brief add not negative finite value
 */
template <typename T>
struct AddNonnegativeFinite{
  __host__ __device__
  T operator()(T x, T y)
  {
    T val = x + y;
    val = (isfinite(val) && (val > 0))? val : T(0);
    return val;
  }
};

/*!
 * @brief calculate background increment
 *  0 backgroundIncrement
 *  1 output_frames/fourierIntensity
 *  2 etai
 *  3 signal
 */
struct BackgroundIncrementCalculator{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple T){
    /* T[0] = signal - (fourierIntensity/etai) */
    thrust::get<0>(T) = thrust::get<3>(T) - (thrust::get<1>(T)/thrust::get<2>(T));
  }
};

/*!
 * @brief compute fourier intensities given parameters
 * 0 output_frames
 * 1 fourierIntensity
 * 2 signal
 * 3 m_frames
 * 4 repeated_mean_bg
 *
 */
struct FourierIntensitySignalCalculator
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple T){
    /* T[1] = fI = |f|^2 */
    /* T[2] = d = m_frames - repeated_mean_bg */
    float t1 = cusp::abs(thrust::get<0>(T))*cusp::abs(thrust::get<0>(T));
    float t2 = thrust::get<3>(T) - thrust::get<4>(T);
    thrust::get<1>(T) = t1;
    thrust::get<2>(T) = t2;
  }

};

/*!
 * @brief return real value
 */
template <typename T1,typename T2>
struct DataProj : public thrust::binary_function<T1,T2,T1>
{
  __host__ __device__
  T1 operator()(T1 x, T2 y){
    if(x != T1(0)){
      return (x/cusp::abs(x))*sqrtf(y);
    }
    return T1(0);
  }
};

/*!
 * @brief return calar fourier intensity given parameters
 *   0 output_frames
 *   1 fourierIntensity
 *   2 m_frames
 *   3 repeated_mean_bg
 */
struct ScaleFourierIntensity{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple T){
    float signalVal = thrust::get<2>(T) - thrust::get<3>(T);
    signalVal = (signalVal < 0)? 0 : signalVal;

    cusp::complex<float> scaledFourierIntensity =
      thrust::get<0>(T)*sqrtf(signalVal/thrust::get<1>(T));
    thrust::get<0>(T) = (isfinite(scaledFourierIntensity.real()) &&
      isfinite(scaledFourierIntensity.imag())) ?
        scaledFourierIntensity : cusp::complex<float>(0,0);
  }
};

/*! Multiply a vector with the conjugate complex of another */
template <typename T>
struct ConjugateMultiply : public thrust::binary_function<T,T,T>
{
  __host__ __device__  T operator()(T x, T y)
  {
#if CUSP_VERSION >= 500
    return x*cusp::detail::conjugate<T>()(y);
#else
    return x*cusp::conj(y);
#endif    
  }
};

/*! Functor that takes the inverse of a vector with regularization */
template <typename T>
struct InvSigma : public thrust::unary_function<T,T>
{
  T sigma;
  InvSigma(T _sigma)
    :sigma(_sigma){}

  __host__ __device__ T operator()(T x){
    return T(1)/(x+sigma);
  }
};

/*!
 * @brief unary absolute value operation
 */
template <typename T>
struct Abs : public thrust::unary_function<T,T>{
  __host__ __device__ T operator()(T x){
#if CUSP_VERSION >= 500
    using thrust::abs;
    using std::abs;
    return abs(x);
#else
    return cusp::abs(x);
#endif
  }
};

/*! Functor to divide two vectors with regularization */
template <typename T>
struct DivideWithRegularization{
  T regularization;
  DivideWithRegularization(T reg) {
      regularization = reg;
  }
  __host__ __device__ T operator()(T numerator, T denominator){
    T val = numerator/(denominator + regularization);
    return val;
  }
};

/*!
 * @brief tiled range class
 */
template <typename Iterator>
class TiledRange{
public:
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  /*!
   * @brief tile struct unary functor
   */
  struct tile_functor : public thrust::unary_function<difference_type,difference_type>{
    difference_type tile_size;
    tile_functor(difference_type tile_size)
      : tile_size(tile_size) {}
    __host__ __device__ difference_type operator()(const difference_type& i) const{
      return i % tile_size;
    }
  };
  typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
  typedef typename thrust::transform_iterator<tile_functor, CountingIterator>   TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;
  // type of the TiledRange iterator
  typedef PermutationIterator iterator;
  // construct repeated_range for the range [first,last)
  TiledRange(Iterator first, Iterator last, difference_type tiles)
    : first(first), last(last), tiles(tiles) {}
  TiledRange(Iterator first, Iterator last, std::vector<int> tile_vec)
    : first(first), last(last), tile_vec(tile_vec), tiles(tile_vec[0]) {
  }

  iterator begin(void) const{
    return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
  }

  iterator end(void) const{
    return begin() + tiles * (last - first);
  }

protected:
  Iterator first;
  Iterator last;
  difference_type tiles;
  std::vector<int> tile_vec;
};

/*! Return the maximum of a cutoff and the division of two vectors */
template <typename T>
struct DivideWithCutoff{
  T cutoff;
  DivideWithCutoff(T c) {
      cutoff = c;
  }
  __host__ __device__ T operator()(T numerator, T denominator){
    T tempEta = numerator/denominator;
    tempEta = ((isfinite((T)tempEta)) && (tempEta >= cutoff))? tempEta : cutoff;
    return tempEta;
  }
};


template <typename T>
struct TupleAdd : public thrust::binary_function<T,T,T>
{
  __host__ __device__
  T operator()(T x, T y)
  { 
    T out;
    thrust::get<0>(out) = thrust::get<0>(x)+thrust::get<0>(y);
    thrust::get<1>(out) = thrust::get<1>(x)+thrust::get<1>(y);
    return out;
  }
};

template <typename IN, typename OUT1, typename OUT2>
struct ComputeImageResidual
{
  ComputeImageResidual() {}
  
  template <typename Tuple>
  __host__ __device__
  thrust::tuple<OUT1,OUT2> operator()(Tuple x)
  { 
    thrust::tuple<OUT1,OUT2> out;

    IN solution = thrust::get<0>(x);
    IN illuminated_area = thrust::get<1>(x);
    IN image = thrust::get<2>(x);

    IN tmp = illuminated_area.real()*conj(solution)*image;
    OUT1 phi = OUT1(tmp.real(), tmp.imag());
    OUT2 image_norm_2 = illuminated_area.real()*cusp::norm(image);
     
    thrust::get<0>(out) = phi;
    thrust::get<1>(out) = image_norm_2;

    return out;
  }
};


}

#endif

