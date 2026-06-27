#ifndef TORCHEDCLIPS_NEURAL_H
#define TORCHEDCLIPS_NEURAL_H

#include <string>
#include <vector>
#include <memory>
#include "TorchedClips_Symbolic.h" // Importa as abstrações do CLIPS (Fact, ClipsValue)

// Forward declaration do LibTorch para manter este header extremamente leve.
// Isso evita que todo arquivo que inclua TorchedClips_Neural.h precise compilar o gigante <torch/torch.h>
namespace torch {
    class Tensor;
}

namespace TorchedClips {

    // 1. Exceção Customizada para a Camada Neural
    class NeuralException : public std::runtime_error {
    public:
        explicit NeuralException(const std::string& message) : std::runtime_error(message) {}
    };

    // 2. O Motor Conexionista Encapsulado (RAII Manager para TorchScript)
    class NeuralModel {
    public:
        NeuralModel();
        ~NeuralModel();

        // Evita cópias (Regra dos 3/5) devido ao gerenciamento do ponteiro interno
        NeuralModel(const NeuralModel&) = delete;
        NeuralModel& operator=(const NeuralModel&) = delete;

        // Carrega um modelo exportado via PyTorch (ex: modelo.pt)
        bool load(const std::string& modelPath);
        bool isLoaded() const;

        // Move o modelo para a GPU (CUDA) se estiver disponível
        void toGPU();

        // Executa a inferência na rede neural (Forward pass)
        // Lança NeuralException em caso de erro no tensor
        torch::Tensor predict(const torch::Tensor& input);

    private:
        // Pimpl Idiom (Pointer to Implementation)
        // Esconde a struct `torch::jit::script::Module` do header público
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };

    // 3. Ponte Neuro-Simbólica (Data Bridge)
    // Classe estática utilitária para converter Tensors em Fatos Simbólicos
    class NeuroSymbolicBridge {
    public:
        // Converte um Tensor 1D do LibTorch em um vetor de ClipsValues
        static std::vector<ClipsValue> tensorToClipsValues(const torch::Tensor& tensor);

        // Cria um Fato estruturado do CLIPS a partir da saída de classificação da RNA
        // Ex: Se prediction conter [0.1, 0.9] e threshold for 0.5, pode extrair a classe dominante.
        static Fact createFactFromPrediction(
            const std::string& templateName, 
            const std::string& slotName, 
            const torch::Tensor& prediction,
            double threshold = 0.5
        );
    };

} // namespace TorchedClips

#endif // TORCHEDCLIPS_NEURAL_H