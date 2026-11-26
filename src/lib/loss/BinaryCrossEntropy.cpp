#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>

namespace dnn {

std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}
double BinaryCrossEntropy::getCostValue(const std::vector<double>& predictedOutput,
                                        const std::vector<double>& actualOutput) const {
 if (predictedOutput.empty()) {
        throw std::invalid_argument("BinaryCrossEntropy: predictedOutput must not be empty");
    }
    if (predictedOutput.size() != actualOutput.size()) {
        throw std::invalid_argument("BinaryCrossEntropy: predictedOutput and actualOutput differ in size");
    }

    const std::size_t n = predictedOutput.size();

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        const double p = clampProbability(predictedOutput[i]);
        const double a = std::log(p);        // P[i]
        const double b = std::log(1.0 - p);  // Q[i]
        const double c = 1.0 - actualOutput[i];  // R[i]
        sum += actualOutput[i] * a + c * b;      // actual . P + R . Q
    }
    return -sum / static_cast<double>(n);
}

double BinaryCrossEntropy::getAccuracyValue(const std::vector<double>& predictedOutput,
                                            const std::vector<double>& actualOutput) const {
if (predictedOutput.empty()) {
        throw std::invalid_argument("BinaryCrossEntropy: predictedOutput must not be empty");
    }
    if (predictedOutput.size() != actualOutput.size()) {
        throw std::invalid_argument("BinaryCrossEntropy: predictedOutput and actualOutput differ in size");
    }    const std::size_t n = predictedOutput.size();

    std::size_t correct = 0;
    for (int i = 0; i < n; i++) {
        const double v = predictedOutput[i] > 0.5 ? 1.0 : 0.0;  // V[i]
        if (v == actualOutput[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(n);
}

std::vector<double> BinaryCrossEntropy::getGradient(const std::vector<double>& predictedOutput,
                                                    const std::vector<double>& actualOutput) const {
if (predictedOutput.empty()) {
        throw std::invalid_argument("BinaryCrossEntropy: predictedOutput must not be empty");
    }
    if (predictedOutput.size() != actualOutput.size()) {
        throw std::invalid_argument("BinaryCrossEntropy: predictedOutput and actualOutput differ in size");
    }    const std::size_t n = predictedOutput.size();

    std::vector<double> gradient(n);
    for (int i = 0; i < n; i++) {
        const double p = clampProbability(predictedOutput[i]);
        gradient[i] = -(actualOutput[i] / p - (1.0 - actualOutput[i]) / (1.0 - p)) /
                      static_cast<double>(n);
    }
    return gradient;
}

std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}


}
