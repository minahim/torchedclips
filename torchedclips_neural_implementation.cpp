#include "TorchedClips_Neural.h"
#include <iostream>

// Inclui os headers pesados do LibTorch apenas no arquivo de implementação
#include <torch/torch.h>
#include <torch/script.h>

namespace TorchedClips {

// --- Implementação do Pimpl (Detalhes Ocultos do LibTorch) ---

class NeuralModel::Impl {
public:
    torch::jit::script::Module module;
    bool isLoaded = false;
    torch::Device device = torch::kCPU; // Padrão é CPU
};

// --- Implementação da Classe NeuralModel ---

NeuralModel::NeuralModel() : m_impl(std::make_unique<Impl>()) {
    // Verifica disponibilidade de CUDA logo na inicialização
    if (torch::cuda::is_available()) {
        std::cout << "[TorchedClips Neural] CUDA disponivel. Considere chamar toGPU()." << std::endl;
    }
}

NeuralModel::~NeuralModel() = default; // Destrutor padrão lida com o unique_ptr

bool NeuralModel::load(const std::string& modelPath) {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        m_impl->module = torch::jit::load(modelPath);
        m_impl->module.to(m_impl->device);
        m_impl->isLoaded = true;
        return true;
    }
    catch (const c10::Error& e) {
        throw NeuralException("Falha ao carregar o modelo PyTorch: " + std::string(e.what()));
    }
}

bool NeuralModel::isLoaded() const {
    return m_impl->isLoaded;
}

void NeuralModel::toGPU() {
    if (torch::cuda::is_available()) {
        m_impl->device = torch::kCUDA;
        if (m_impl->isLoaded) {
            m_impl->module.to(m_impl->device);
        }
    } else {
        std::cerr << "[TorchedClips Neural Aviso] CUDA nao esta disponivel. Permanecendo na CPU." << std::endl;
    }
}

torch::Tensor NeuralModel::predict(const torch::Tensor& input) {
    if (!m_impl->isLoaded) {
        throw NeuralException("Tentativa de inferencia sem modelo carregado.");
    }

    try {
        // Move a entrada para o mesmo device do modelo (CPU/GPU)
        torch::Tensor deviceInput = input.to(m_impl->device);
        
        // Cria o vetor de entradas esperado pelo TorchScript
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(deviceInput);

        // Executa o forward pass e extrai o tensor resultante
        torch::Tensor output = m_impl->module.forward(inputs).toTensor();
        
        // Retorna o resultado sempre para a CPU para facilitar a manipulação com CLIPS
        return output.to(torch::kCPU);
    }
    catch (const c10::Error& e) {
        throw NeuralException("Erro durante a execucao do forward pass: " + std::string(e.what()));
    }
}

// --- Implementação da Ponte Neuro-Simbólica ---

std::vector<ClipsValue> NeuroSymbolicBridge::tensorToClipsValues(const torch::Tensor& tensor) {
    std::vector<ClipsValue> result;
    
    // Garante que estamos lidando com tensores 1D achatados e na CPU
    torch::Tensor flatTensor = tensor.flatten().to(torch::kCPU);
    
    // Verifica o tipo do tensor para fazer o cast correto
    if (flatTensor.dtype() == torch::kFloat32 || flatTensor.dtype() == torch::kFloat64) {
        auto data_ptr = flatTensor.data_ptr<float>();
        for (int i = 0; i < flatTensor.numel(); ++i) {
            result.push_back(static_cast<double>(data_ptr[i]));
        }
    } else if (flatTensor.dtype() == torch::kInt32 || flatTensor.dtype() == torch::kInt64) {
        auto data_ptr = flatTensor.data_ptr<int64_t>();
        for (int i = 0; i < flatTensor.numel(); ++i) {
            result.push_back(static_cast<long long>(data_ptr[i]));
        }
    } else {
        throw NeuralException("Tipo de tensor nao suportado na conversao para ClipsValue.");
    }
    
    return result;
}

Fact NeuroSymbolicBridge::createFactFromPrediction(
    const std::string& templateName, 
    const std::string& slotName, 
    const torch::Tensor& prediction,
    double threshold) 
{
    // Supomos que prediction e' um tensor de probabilidades (ex: softmax ou sigmoid)
    Fact fact(templateName);
    
    // Usamos Argmax para encontrar a classe mais provavel
    torch::Tensor class_idx = prediction.argmax();
    long long class_id = class_idx.item<long long>();
    
    // Pega a probabilidade associada
    double prob = prediction[0][class_id].item<double>(); // Assumindo batch_size = 1
    
    // Define os valores no Fato CLIPS
    fact.setSlot("id_classe", static_cast<long long>(class_id));
    fact.setSlot("confianca", static_cast<double>(prob));
    
    // Se a probabilidade for maior que o threshold, marca como verificado
    fact.setSlot("verificado", prob >= threshold);
    
    return fact;
}

} // namespace TorchedClips