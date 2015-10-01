#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HuberLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
    
  caffe_gpu_powx(count, diff_.gpu_data(), Dtype(2.0), huber_.mutable_gpu_data());
  caffe_gpu_add_scalar(count, Dtype(1.0), huber_.mutable_gpu_data());
  caffe_gpu_powx(count, huber_.gpu_data(), Dtype(0.5), huber_.mutable_gpu_data());  

  Dtype sum;
  caffe_gpu_asum(count, diff_.gpu_data(), &sum);  
  
  Dtype loss = sum / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	caffe_gpu_div(bottom[0]->count(), diff_.gpu_data(), huber_.gpu_data(), diff_.mutable_gpu_data());

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();

      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a 
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HuberLossLayer);

}  // namespace caffe
