#ifndef LIBTORCHEDCLIPS_H
#define LIBTORCHEDCLIPS_H

#include "libtorchedclips_global.h"


// Encapsulamento de tipos para evitar vazamento de headers pesados nas aplicações cliente
namespace torch { class Tensor; }

#endif // LIBTORCHEDCLIPS_H
class LIBTORCHEDCLIPS_EXPORT Libtorchedclips
{
public:
    Libtorchedclips();
};

#endif // LIBTORCHEDCLIPS_H
