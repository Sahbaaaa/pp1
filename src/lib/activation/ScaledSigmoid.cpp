#include <activation/ScaledSigmoid.h>
namespace dnn {
    double ScaledSigmoid::activation (double x) {
        return min + (max-min) * ((1.0) / (1.0 + std::exp (-x) ));
    }
    double ScaledSigmoid::activationBackwards (double x) {
       
return (max-min) * (1/ (1+ std::exp (-x))) * (1 -(1/ (1+ std::exp (-x))));
       
}
std::string ScaledSigmoid :: getName() const
{
    return "scaled_sigmoid";
}
double ScaledSigmoid :: getMin() const
{
    return min;
}
double ScaledSigmoid :: getMax() const
{
    return max;
}
}
