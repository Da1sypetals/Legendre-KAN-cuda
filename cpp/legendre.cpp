#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void leg_launcher(const float *x, float *leg, int batch_size, int in_feats, int degree);
void leg_bwd_launcher(const float *gout, const float *x, const float *leg, float *grad_x, int batch_size, int in_feats, int degree);

torch::Tensor leg_cuda_fwd(torch::Tensor x, int degree)
{

    CHECK_INPUT(x);

    const float *x_ptr = x.data_ptr<float>();
    int batch_size = x.size(0);
    int in_feats = x.size(1);

    // create leg tensor
    torch::Tensor leg = torch::ones({degree + 1, batch_size, in_feats},
                                    torch::device(torch::kCUDA).dtype(torch::kFloat));

    float *leg_ptr = leg.data_ptr<float>();

    leg_launcher(x_ptr, leg_ptr, batch_size, in_feats, degree);

    return leg;
}

torch::Tensor leg_cuda_bwd(torch::Tensor gout, torch::Tensor x, torch::Tensor leg)
{

    CHECK_INPUT(x);
    CHECK_INPUT(leg);

    const float *gout_ptr = gout.data_ptr<float>();
    const float *x_ptr = x.data_ptr<float>();
    const float *leg_ptr = leg.data_ptr<float>();

    int batch_size = x.size(0);
    int in_feats = x.size(1);
    int degree = leg.size(0) - 1;

    // create grad_x tensor
    torch::Tensor grad_x = torch::zeros_like(x);
    float *grad_x_ptr = grad_x.data_ptr<float>();

    leg_bwd_launcher(gout_ptr, x_ptr, leg_ptr, grad_x_ptr, batch_size, in_feats, degree);

    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &leg_cuda_fwd, "leg forward");
    m.def("backward", &leg_cuda_bwd, "leg backward");
}