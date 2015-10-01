#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HuberLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  huber_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());    
    
  caffe_powx(count, diff_.cpu_data(), Dtype(2.0), huber_.mutable_cpu_data());
  caffe_add_scalar(count, Dtype(1.0), huber_.mutable_cpu_data());
  caffe_powx(count, huber_.cpu_data(), Dtype(0.5), huber_.mutable_cpu_data());  
  
  Dtype sum = caffe_cpu_asum(count, diff_.cpu_data());

  Dtype loss = sum / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	caffe_div(bottom[0]->count(), diff_.cpu_data(), huber_.cpu_data(), diff_.mutable_cpu_data());	

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
	  
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
		  diff_.cpu_data(),                   // a  
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HuberLossLayer);
#endif

INSTANTIATE_CLASS(HuberLossLayer);
REGISTER_LAYER_CLASS(HuberLoss);

}  // namespace caffe
