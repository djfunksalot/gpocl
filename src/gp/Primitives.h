// -----------------------------------------------------------------------------
// $Id$
//
//   Primitives.h
// 
//   Genetic Programming in OpenCL (gpocl)
//
//   Copyright (C) 2010-2010 Douglas A. Augusto
// 
// This file is part of gpocl
// 
// GPOCL is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
// 
// GPOCL is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
// 
// You should have received a copy of the GNU General Public License along
// with GPOCL; if not, see <http://www.gnu.org/licenses/>.
//
// -----------------------------------------------------------------------------

#ifndef _primitives_h
#define _primitives_h

#include <CL/cl.hpp>

#include "../common/Exception.h"
#include "../common/util/Util.h"

// Functions definition
#include "kernels/common.cl"

#include <string>
#include <iostream>
#include <vector>
#include <cassert>

// -----------------------------------------------------------------------------
class Primitives {
public:
   /**
    * @class Error
    *
    * @brief Class for GP's fatal exceptions.
    */
   struct Error: public Exception {
      Error( const std::string& msg ): Exception( "@ Primitives ", msg ) {};
   };

   enum { GPT_EPHEMERAL, GPF_IDENTITY, GPT_VAR = 127 };
public:
//   Primitives();

   struct Primitive { 
      Primitive( cl_uint a, const std::string& n, const std::string& s, const std::string& c ):
         arity( a ), name( n ), symbol( s ), code( c ) {}
      cl_uint arity;
      std::string name;
      std::string symbol;
      std::string code;
   };

   /** @brief The database of primitives

     Each member of DB holds the name, symbol (alias), arity, and type of the primitive.
    */
   std::vector<Primitive> DB;
   std::vector<cl_uint> m_primitives;

public:

   void ShowAvailablePrimitives() const { /* TODO */ }
   cl_uint RandomGene( unsigned min, unsigned max );
   void Load( unsigned, unsigned, const std::string& );

private:
   /**
     Try to find the corresponding primitive by name or symbol. When it finds,
     then it returns a pair of 'arity' and 'index'. Otherwise it throws an error.
     */
   std::pair<cl_uint, cl_uint> Find( const std::string& token );
   void Register( cl_uint, const std::string&, const std::string&, const std::string& );

   std::vector<std::pair<unsigned, unsigned> > m_primitives_boundaries;
   unsigned m_max_arity;

private:
   // --------------
   cl_uint PackAlelo( cl_uint arity, cl_uint type ) {
      assert( sizeof(cl_uint) == 4 );

      // checking bounds
      assert( ! (arity & 0xFFFFFFF8) ); // 0xFFFFFFF8 = 11111111 11111111 11111111 11111000
      assert( ! (type  & 0xFFFFFF80) ); // 0xFFFFFF80 = 11111111 11111111 11111111 10000000

      return (arity << 29) | (type << 22);
      //  return PACKALELO( arity, type );
   }
   cl_uint PackAlelo( cl_uint arity, cl_uint type, cl_uint index ) {
      // checking bounds
      assert( ! (index & 0xFFC00000) ); // 0xFFC00000 = 11111111 11000000 00000000 00000000

      return PackAlelo( arity, type ) | index;
   }
   cl_uint PackAlelo( cl_uint arity, cl_uint type, cl_float value ) {
      unsigned packed_value = util::RndPosNum<unsigned>( value * COMPACT_RANGE / 1024 );
      // Checking bounds, i.e. can packed_value fit in 22 bits?)
      assert( ! (packed_value & 0xFFC00000) ); // 0xFFC00000 = 11111111 11000000 00000000 00000000

      return PackAlelo( arity, type ) | packed_value;
   }
   // --------------
};

// -----------------------------------------------------------------------------
#endif
