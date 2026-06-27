#include "TorchedClips_Symbolic.h"
#include <iostream>
#include <sstream>

// Importação da biblioteca C do CLIPS
// Isolado no arquivo .cpp para não poluir quem inclui o .h
extern "C" {
    #include "clips.h"
}

namespace TorchedClips {

// --- Implementação da Classe Fact ---

Fact::Fact(const std::string& templateName) : m_templateName(templateName) {}

void Fact::setSlot(const std::string& slotName, const ClipsValue& value) {
    m_slots[slotName] = value;
}

void Fact::addToList(const ClipsValue& value) {
    m_orderedValues.push_back(value);
}

std::string Fact::getTemplateName() const {
    return m_templateName;
}

std::string Fact::formatValue(const ClipsValue& val) const {
    if (std::holds_alternative<long long>(val)) return std::to_string(std::get<long long>(val));
    if (std::holds_alternative<double>(val)) return std::to_string(std::get<double>(val));
    if (std::holds_alternative<bool>(val)) return std::get<bool>(val) ? "TRUE" : "FALSE";
    
    // Para strings, verifica se precisa de aspas (se tiver espaço) ou se é um symbol
    std::string str = std::get<std::string>(val);
    if (str.find(' ') != std::string::npos) return "\"" + str + "\"";
    return str;
}

std::string Fact::toClipsString() const {
    std::ostringstream oss;
    oss << "(" << m_templateName;
    
    // Fatos ordenados (implied facts) ex: (dado 1 2 3)
    if (!m_orderedValues.empty()) {
        for (const auto& val : m_orderedValues) {
            oss << " " << formatValue(val);
        }
    } 
    // Fatos estruturados (deftemplates) ex: (pessoa (nome "Ana") (idade 30))
    else {
        for (const auto& pair : m_slots) {
            oss << " (" << pair.first << " " << formatValue(pair.second) << ")";
        }
    }
    
    oss << ")";
    return oss.str();
}

// --- Implementação da Classe Environment ---

Environment::Environment() {
    // Inicializa o ambiente CLIPS. Em CLIPS 6.4+, CreateEnvironment retorna Environment*
    m_env = CreateEnvironment();
    if (!m_env) {
        throw ClipsException("Falha ao inicializar o ambiente CLIPS.");
    }
}

Environment::~Environment() {
    if (m_env) {
        DestroyEnvironment(m_env);
        m_env = nullptr;
    }
}

void Environment::reset() {
    EnvReset(m_env);
}

void Environment::clear() {
    EnvClear(m_env);
}

long long Environment::run(long long limit) {
    // Roda a agenda de regras. Retorna o número de regras executadas.
    return EnvRun(m_env, limit);
}

bool Environment::load(const std::string& filepath) {
    // EnvLoad retorna 1 (true) se sucesso, 0 (false) se falha
    int success = EnvLoad(m_env, filepath.c_str());
    return success != 0;
}

void Environment::build(const std::string& constructString) {
    int success = EnvBuild(m_env, constructString.c_str());
    if (success == 0) {
        throw ClipsException("Falha ao compilar o construct: " + constructString);
    }
}

void Environment::assertString(const std::string& factString) {
    void* factPtr = EnvAssertString(m_env, factString.c_str());
    if (!factPtr) {
        throw ClipsException("Falha ao inserir o fato: " + factString + ". Verifique se o template existe.");
    }
}

void Environment::assertFact(const Fact& fact) {
    assertString(fact.toClipsString());
}

// --- Sistema de Callbacks Bidirecional ---

// Esta estrutura atua como uma "ponte" que guardamos na memória do CLIPS
struct CallbackUserData {
    Environment::ClipsCallback cppFunction;
    std::string functionName;
};

void Environment::registerFunction(const std::string& functionName, ClipsCallback callback) {
    // Alocamos os dados do usuário para persistir durante a vida do ambiente
    auto* userData = new CallbackUserData{callback, functionName};
    
    // Registra a função no CLIPS
    // AddUDF (User Defined Function) é a API moderna do CLIPS 6.4 para funções customizadas
    // Passamos routerCallback como a função C que o CLIPS vai chamar.
    AddUDF(m_env, 
           functionName.c_str(), 
           "v",               // 'v' significa tipo de retorno variável (unknown/any)
           0, 10,             // Min/Max de argumentos
           "v",               // Tipos de argumentos aceitos (v = qualquer)
           (void (*)(Environment*, UDFContext*, UDFValue*))&Environment::routerCallback, 
           "routerCallback", 
           userData);         // Passamos a nossa std::function empacotada no context
}

// A função estática que recebe o roteamento bruto do C puro
void Environment::routerCallback(void* clipsEnv, void* context) {
    // 1. Recuperamos nossa struct C++ do contexto
    UDFContext* udfContext = static_cast<UDFContext*>(context);
    CallbackUserData* userData = static_cast<CallbackUserData*>(udfContext->context);
    
    // (Opcional) Aqui você iteraria sobre UDFArgumentCount() para ler os argumentos
    // convertendo DATA_OBJECT/UDFValue para std::vector<ClipsValue> args
    std::vector<ClipsValue> args; 
    
    // 2. Chamamos a lambda/método C++
    try {
        ClipsValue result = userData->cppFunction(args);
        
        // 3. (Opcional) Converter 'result' de volta para UDFValue do CLIPS para retornar
        // UDFReturnValue(udfContext, retDataObject);
        
    } catch (const std::exception& e) {
        std::cerr << "[TorchedClips Error in Rule Callback]: " << e.what() << std::endl;
    }
}

} // namespace TorchedClips