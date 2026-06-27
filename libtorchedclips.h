#ifndef LIBTORCHEDCLIPS_H
#define LIBTORCHEDCLIPS_H

#include "libtorchedclips_global.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <memory>
#include <stdexcept>
#include <functional>

// Encapsulamento de tipos para evitar vazamento de headers pesados nas aplicações cliente
namespace torch { class Tensor; }

namespace TorchedClips {

// 1. Abstração de Tipos de Dados Dinâmicos do CLIPS
// O CLIPS é tipado dinamicamente. std::variant é a forma C++ moderna de lidar com DATA_OBJECT.
using ClipsValue = std::variant<
    long long,          // INTEGER
    double,             // FLOAT
    std::string,        // SYMBOL ou STRING
    bool                // Utilizado para retornos lógicos
    >;

// 2. Abstração de um Fato (Fact) Orientado a Objetos
class Fact {
public:
    Fact(const std::string& templateName);

    void setSlot(const std::string& slotName, const ClipsValue& value);
    void addToList(const ClipsValue& value); // Para multifields ou implied facts

    std::string getTemplateName() const;
    std::string toClipsString() const; // Converte o objeto para a sintaxe (template (slot valor))

private:
    std::string m_templateName;
    std::unordered_map<std::string, ClipsValue> m_slots;
    std::vector<ClipsValue> m_orderedValues;

    std::string formatValue(const ClipsValue& val) const;
};

// 3. Exceção Customizada para erros do CLIPS
class ClipsException : public std::runtime_error {
public:
    explicit ClipsException(const std::string& message) : std::runtime_error(message) {}
};

// 4. O Motor Simbólico Encapsulado (RAII Manager)
class Environment {
public:
    // Construtor inicializa o ambiente CLIPS (CreateEnvironment)
    Environment();

    // Destrutor libera a memória (DestroyEnvironment)
    ~Environment();

    // Evita cópias acidentais do ponteiro de ambiente (Regra dos 3/5)
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    // Comandos de Controle Básicos
    void reset();
    void clear();
    long long run(long long limit = -1); // Retorna o número de regras disparadas

    // Carregamento e Inserção de Conhecimento
    bool load(const std::string& filepath);
    void build(const std::string& constructString);
    void assertFact(const Fact& fact);
    void assertString(const std::string& factString);

    // Integração Avançada: Registra uma função C++ para ser chamada de dentro de uma regra CLIPS
    // Isso é crucial para o Fluxo Reverso (Simbólico -> Neuro)
    using ClipsCallback = std::function<ClipsValue(const std::vector<ClipsValue>& args)>;
    void registerFunction(const std::string& functionName, ClipsCallback callback);

private:
    void* m_env; // Ponteiro opaco (void*) para esconder o header <clips.h> do resto do projeto C++

    // Ponteiro estático interno usado para rotear as chamadas C para o std::function C++
    static void routerCallback(void* clipsEnv, void* context);
};

// --- Tipos de Dados Abstratos (Abstractions) ---

// Representação OO de um fato simbólico pronto para o CLIPS
struct SymbolicFact {
    std::string relation;
    std::unordered_map<std::string, std::string> slots; // Para deftemplates (fatos estruturados)
    std::vector<std::string> orderedValues;             // Para fatos ordenados simples
};

// Classe utilitária para conversão segura entre Tensores numéricos e Fatos simbólicos
class DataBridge {
public:
    static SymbolicFact tensorToFact(const std::string& relationName, const torch::Tensor& tensor, double threshold = 0.5);
    static torch::Tensor factToTensor(const SymbolicFact& fact);
};

// --- Componentes Core da Interface Neuro-Simbólica ---

// Interface para gerenciamento de modelos do LibTorch (Lado Neuro)
class NeuralModule {
public:
    NeuralModule(const std::string& modelPath);
    ~NeuralModule();

    torch::Tensor forward(const torch::Tensor& inputTensor);
    bool isCudaAvailable() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl; // Idioma Pimpl para isolar dependências do LibTorch
};

// Interface para gerenciamento do ambiente CLIPS (Lado Simbólico)
class SymbolicEnvironment {
public:
    SymbolicEnvironment();
    ~SymbolicEnvironment();

    bool loadRules(const std::string& filePath);
    void assertFact(const SymbolicFact& fact);
    void assertRawFactString(const std::string& factString);
    void runRules();

    // Registra uma função C++ que o CLIPS pode chamar de dentro das regras
    void registerCallback(const std::string& clipsFunctionName,
                          void* context,
                          void (*callbackFunction)(void* env, void* context));

private:
    void* clipsEnvPtr; // Ponteiro opaco para a instância do ambiente CLIPS (Environment)
};

// --- O Orquestrador Bidirecional (Interface Principal) ---

class NeuroSymbolicBridge {
public:
    NeuroSymbolicBridge(const std::string& neuralModelPath, const std::string& clipsRulesPath);
    ~NeuroSymbolicBridge();

    // Ciclo Executivo 1: Entrada de dados -> IA Neural -> Fatos no CLIPS -> Inferência Simbólica
    void executeNeuralToSymbolicFlow(const torch::Tensor& rawSensoryData);

    // Acesso direto aos subsistemas se necessário
    SymbolicEnvironment& getSymbolicCore() { return *symbolicCore; }
    NeuralModule& getNeuralCore() { return *neuralCore; }

private:
    std::unique_ptr<NeuralModule> neuralCore;
    std::unique_ptr<SymbolicEnvironment> symbolicCore;

    // Função estática interna que serve como o ponto de entrada para o CLIPS chamar a rede neural
    static void clipsToNeuralCallback(void* env, void* context);
};

} // namespace TorchedClips

#endif // LIBTORCHEDCLIPS_H
class LIBTORCHEDCLIPS_EXPORT Libtorchedclips
{
public:
    Libtorchedclips();
};

#endif // LIBTORCHEDCLIPS_H
