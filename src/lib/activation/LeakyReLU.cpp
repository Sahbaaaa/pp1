#include <activation/LeakyReLU.h>
namespace dnn {
    double LeakyReLU::activation (double x) {
        if (x> 0.0)
        {
            return x;
        }
        return 0.01 * x;
    }
    double LeakyReLU::activationBackwards (double x) {
        if (x>0.0)
        { return 1.0;
}
return 0.01;
       
}
std::string LeakyReLU :: getName() const
{
    return " leaky_relu";
}
}