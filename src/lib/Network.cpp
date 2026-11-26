#include <Network.h>

#include <iostream>
#include <stdexcept>

#include <string>
#include <tuple>


namespace dnn {
    Network::Network(
    std::vector<std::tuple<
        std::vector<std::vector<double>>,
        std::vector<double>,
        ActivationFunction*
    >> layerParameters,
    LossFunction* lossFunction
)
    : ownedLossFunction(lossFunction) {

    for (auto& params : layerParameters) {

        std::vector<std::vector<double>>& weights = std::get<0>(params);
        std::vector<double>& bias = std::get<1>(params);
        ActivationFunction* activationFunction = std::get<2>(params);

        if (weights.empty()) {
            throw std::invalid_argument("Layer weights must not be empty");
        }

        const uint64_t outputSize = weights.size();
        const uint64_t inputSize = weights[0].size();

        if (bias.size() != outputSize) {
            throw std::invalid_argument("Bias size does not match output size");
        }

        for (const auto& row : weights) {
            if (row.size() != inputSize) {
                throw std::invalid_argument("Inconsistent weight matrix dimensions");
            }
        }

        Layer* layer = new Layer(
            layers.size(),
            nextSeed,
            inputSize,
            outputSize,
            activationFunction,
            "Layer_" + std::to_string(layers.size())
        );

        layer->setParameters(weights, bias);

        layers.push_back(layer);
        nextSeed++;
    }
}

Network::~Network() {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers.clear();
    delete ownedLossFunction;
    ownedLossFunction = nullptr;
}
Network& Network::addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction* activationFunction) {
    Layer* layer = new Layer(layers.size(), nextSeed, inputDim, outputDim, activationFunction);
    layer->init();
    layers.push_back(layer);
    nextSeed++;
    return *this;
}

const std::vector<double>& Network::getLayerInput(uint64_t layerId) const {
    if (layerId >= layers.size()) {
        throw std::out_of_range("Layer id " + std::to_string(layerId) + " does not exist");
    }
    if (layerId == 0) {
        return initialInput;
    }
    return layers[layerId - 1]->getActivationOutput();
}

void Network::fullForwardPropagation(const std::vector<double>& input) {
    if (layers.empty()) {
        throw std::logic_error("Network has no layers");
    }
    initialInput = input;

    for (int i = 0; i < layers.size(); i++) {
        layers[i]->forwardPropagation(getLayerInput(i));
    }
    output = layers.back()->getActivationOutput();
}

void Network::fullBackwardPropagation(const std::vector<double>& predictedOutput,
                                      const std::vector<double>& actualOutput) {
    if (ownedLossFunction == nullptr) {
        throw std::logic_error("Network has no loss function");
    }
    if (layers.empty()) {
        throw std::logic_error("Network has no layers");
    }

    std::vector<double> dACurr = ownedLossFunction->getGradient(predictedOutput, actualOutput);

    for (int i = layers.size(); i-- > 0;) {
        layers[i]->backwardPropagation(dACurr, getLayerInput(i));
        dACurr = layers[i]->getDPreviousLayerOutput();
    }
}

void Network::trainBatch(const std::vector<std::vector<double>>& inputs,
                         const std::vector<std::vector<double>>& actualOutputs,
                         std::size_t batchSize, int epochs, double learningRate) {
    if (ownedLossFunction == nullptr) {
        throw std::logic_error("trainBatch: no loss function has been set");
    }
    if (layers.empty()) {
        throw std::logic_error("trainBatch: network has no layers");
    }
    if (inputs.empty() || actualOutputs.empty()) {
        throw std::invalid_argument("trainBatch: inputs and actualOutputs must not be empty");
    }
    if (inputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("trainBatch: inputs and actualOutputs differ in size");
    }
    if (batchSize == 0) {
        throw std::invalid_argument("trainBatch: batchSize must be greater than 0");
    }
    const std::size_t inputSize = layers.front()->getWeights()[0].size();
    const std::size_t outputSize = layers.back()->getWeights().size();
    for (std::size_t s = 0; s < inputs.size(); s++) {
        if (inputs[s].size() != inputSize) {
            throw std::invalid_argument("trainBatch: input " + std::to_string(s) +
                                        " does not match the input size of the first layer");
        }
        if (actualOutputs[s].size() != outputSize) {
            throw std::invalid_argument("trainBatch: actual output " + std::to_string(s) +
                                        " does not match the output size of the last layer");
        }
    }

    costHistory.clear();
    accuracyHistory.clear();

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epochCost = 0.0;
        double epochAccuracy = 0.0;
        std::size_t processed = 0;

        for (std::size_t s = 0; s < inputs.size(); s++) {
            fullForwardPropagation(inputs[s]);

            epochCost += ownedLossFunction->getCostValue(output, actualOutputs[s]);
            epochAccuracy += ownedLossFunction->getAccuracyValue(output, actualOutputs[s]);

            fullBackwardPropagation(output, actualOutputs[s]);
            processed++;

            if (processed % batchSize == 0) {
                for (Layer* layer : layers) {
                    layer->commitToWeights(learningRate);
                }
            }
        }
        if (processed % batchSize != 0) {
            for (Layer* layer : layers) {
                layer->commitToWeights(learningRate);
            }
        }

        costHistory.push_back(epochCost / static_cast<double>(inputs.size()));
        accuracyHistory.push_back(epochAccuracy / static_cast<double>(inputs.size()));
    }
}

double Network::computeAccuracy(const std::vector<std::vector<double>>& inputs,
                                const std::vector<std::vector<double>>& actualOutputs) {
    if (ownedLossFunction == nullptr) {
        throw std::logic_error("computeAccuracy: no loss function has been set");
    }
    if (layers.empty()) {
        throw std::logic_error("computeAccuracy: network has no layers");
    }
    if (inputs.empty() || actualOutputs.empty()) {
        throw std::invalid_argument("computeAccuracy: inputs and actualOutputs must not be empty");
    }
    if (inputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("computeAccuracy: inputs and actualOutputs differ in size");
    }
    const std::size_t inputSize = layers.front()->getWeights()[0].size();
    for (std::size_t s = 0; s < inputs.size(); s++) {
        if (inputs[s].size() != inputSize) {
            throw std::invalid_argument("computeAccuracy: input " + std::to_string(s) +
                                        " does not match the input size of the first layer");
        }
    }

    double totalCorrect = 0.0;
    for (std::size_t s = 0; s < inputs.size(); s++) {
        fullForwardPropagation(inputs[s]);
        totalCorrect += ownedLossFunction->getAccuracyValue(output, actualOutputs[s]);
    }
    return totalCorrect / static_cast<double>(inputs.size());
}

void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

const LossFunction* Network::getLossFunction() const {
    return ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return output;
}

const std::vector<double>& Network::getCostHistory() const {
    return costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers = std::move(newLayers);
    initialInput.clear();
    output.clear();
    costHistory.clear();
    accuracyHistory.clear();
}

}
