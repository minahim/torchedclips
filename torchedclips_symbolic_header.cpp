#ifndef TORCHEDCLIPS_SYMBOLIC_H
#define TORCHEDCLIPS_SYMBOLIC_H

#include <string>
#include <vector>
#include <variant>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <functional>

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

} // namespace TorchedClips

#endif // TORCHEDCLIPS_SYMBOLIC_H