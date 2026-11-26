#include <MathUtils.h>

#include <stdexcept>
#include <string>

namespace dnn {

double dotProduct(const std::vector<double>& a, const std::vector<double>& b)
{
    double dotProduct = 0.0; 
for (int i =0; i <= a.size() -1; i ++)
{
 dotProduct += (a[i] * b[i]);
}
 return dotProduct;
}

double sum(const std::vector<double>& values)
{
    double sum = 0.0;
for (int i =0; i <= values.size() -1; i ++){
 sum += values[i];
}
return sum;
}

template<typename T>
 std::vector<double> normalize(const std::vector<T>& values)
 {
  T max  = *std::max_element(values.begin(), values.end());

  return normalize (values, max);
 }

 template<typename T>
std::vector<double> normalize(const std::vector<T>& values, int maxValue)
{
    std::vector<double> val (values.size());
    for( int i =0; i < values.size(); i++)
  {
   val[i] = values[i]/maxValue;
  }
  return val;

}

std::vector<double> encodeAsOneHot(double value, std::size_t numClasses) {
    if (value < 0.0 || value >= static_cast<double>(numClasses)) {
        throw std::out_of_range("encodeAsOneHot: value " + std::to_string(value) + " is out of range for " +
                                std::to_string(numClasses) + " classes");
    }

    std::vector<double> encoded(numClasses, 0.0);
    encoded[static_cast<std::size_t>(value)] = 1.0;
    return encoded;
}

}
