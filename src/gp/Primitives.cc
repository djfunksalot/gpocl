// -----------------------------------------------------------------------------
// $Id$
//
//   Primitives.cc
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

#include "Primitives.h"

#include <algorithm>

#include "../common/util/Random.h"


// -----------------------------------------------------------------------------
std::pair<cl_uint, cl_uint> Primitives::Find( const std::string& token )
{
   for( unsigned i = 0; i < DB.size(); ++i )
      if( DB[i].name == token || DB[i].symbol == token )
         return std::make_pair( DB[i].arity, i );

   throw Error( "Unrecognized primitive: " + std::string( token ) );
}

// -----------------------------------------------------------------------------
void Primitives::Register( cl_uint arity, const std::string& name, 
                           const std::string& symbol, const std::string& code )
{
   assert( DB.size() < 127 ); // 127 = 2^7 - 1
   DB.push_back( Primitive( arity, name, symbol, code ) );
}

// -----------------------------------------------------------------------------
void Primitives::Load( unsigned x_dim, unsigned max_gen_size, const std::string& primitives )
{
   /* The first two primitives are special, they need to be the first ones. */
   Register( 0, "ephemeral",    "",             "AS_FLOAT( GENE )" );
   assert( GPT_EPHEMERAL == DB.size() - 1 );
   Register( 1, "identity",     "_",            "POP" );
   assert( GPF_IDENTITY == DB.size() - 1 );

   Register( 3, "ifthenelse",   "ITE",          "POP ? POP : (POP, POP)" );

   Register( 2, "add",          "+",            "POP + POP" );
   Register( 2, "minus",        "-",            "POP - POP" );
   Register( 2, "mul",          "*",            "POP * POP" );
   Register( 2, "div",          "/",            "POP / (TOP == 0.0f ? (POP, 1.0f) : POP)" );
   Register( 2, "less",         "<",            "POP < POP" );
   Register( 2, "greater",      ">",            "POP > POP" );
   Register( 2, "lessequal",    "<=",           "POP <= POP" );
   Register( 2, "greaterequal", ">=",           "POP >= POP" );
   Register( 2, "and",          "&&",           "POP && POP" );
   Register( 2, "or",           "||",           "POP || POP" );

   Register( 1, "not",          "!",            "!POP" );
   Register( 1, "neg",          "neg",          "-POP" );
   Register( 1, "sqrt",         "sqrt",         "(TOP < 0.0f ? (POP, 1.0f) : sqrt(POP))" );

   Register( 0, "c_pi",         "3.1415",       "M_PI_F" );
   Register( 0, "c_pi_2",       "1.5707",       "M_PI_2_F" );
   Register( 0, "c_pi_4",       "0.7853",       "M_PI_4_F" );
   Register( 0, "c_1_pi",       "0.3183",       "M_1_PI_F" );
   Register( 0, "c_2_pi",       "0.6366",       "M_2_PI_F" );
   Register( 0, "c_2_sqrtpi",   "1.1283",       "M_2_SQRTPI_F" );
   Register( 0, "c_sqrt2",      "1.4142",       "M_SQRT2_F" );
   Register( 0, "c_sqrt1_2",    "0.7071",       "M_SQRT1_2_F" );
   Register( 0, "c_e",          "2.7182",       "M_E_F" );
   Register( 0, "c_log2e",      "1.4426",       "M_LOG2E_F" );
   Register( 0, "c_log10e",     "0.4342",       "M_LOG10E_F" );
   Register( 0, "c_ln2",        "1.6931",       "M_LN2_F" );
   Register( 0, "c_ln10",       "3.3025",       "M_LN10_F" );
   Register( 0, "c_0",          "0",            "0.0f" );
   Register( 0, "c_1",          "1",            "1.0f" );
   Register( 0, "c_2",          "2",            "2.0f" );
   Register( 0, "c_-1",         "-1",           "-1.0f" );
   Register( 0, "c_-2",         "-2",           "-2.0f" );
   Register( 0, "c_apery",      "1.2020",       "1.202056903159594f" );
   Register( 0, "c_catalan",    "0.9159",       "0.915965594177219f" );
   Register( 0, "c_euler",      "0.5772",       "0.5772156649015329f" );
   Register( 0, "c_golden",     "1.6180",       "1.618033988749895f" );
   Register( 0, "c_omega",      "0.5671",       "0.5671432904097839f" );

   /////////////////////////////////////////////////////////////

   std::vector<std::vector<cl_uint> > per_arity;

   // Adding all problem variables (arity = 0)
   per_arity.resize(1);
   for( unsigned i = 0; i < x_dim; ++i )
      per_arity[0].push_back( PackAlelo( 0, GPT_VAR, (cl_uint)i ) );

   // Adding user given primitives (functions and terminals)
   std::stringstream ss( primitives ); std::string token;
   while( std::getline( ss, token, ',' ) )
   {
      // gene.first -> arity, gene.second -> type
      std::pair<cl_uint, cl_uint> gene = Find( token );
      if( gene.first >= per_arity.size() ) per_arity.resize( gene.first + 1 );
      per_arity[gene.first].push_back( PackAlelo( gene.first, gene.second ) );
   }

   // --------------- Consistence check
   if( per_arity.size() > 1 )
   {
   /*
      In order to initialize a linear tree of exactly a given size we need to have
      at least an operator requiring *one* argument. If the user didn't give this
      type of operator, we resort to a fake one-argument function called "identity".
      This function, as one can guess, just returns the value of its operator.
    */
      if( per_arity[1].empty() ) per_arity[1].push_back( PackAlelo( 1, GPF_IDENTITY ) );
   }
   else if( per_arity.size() == 1 && max_gen_size > 1 )
   {
   /*
      The only case where we can allow not having a primitive operator is when
      the maximum genome size is equal to 1, so any terminal can serve any
      genome.
    */
      throw Error( "You should enter at least an operator (function)." );
   }
   // --------------- 

   /* 
      Populate the array of primitives, where in each position is stored a primitive.
      The primitives are ordered by arity, ranging from 0 (terminals) to 'n'.

      Simultaneously, the array of boundaries is filled. It stores the 'start index'
      and 'end index' of each set of arities. This allows us to randomly select
      primitives of a given arity of range of arities.
    */
   m_primitives_boundaries.resize( per_arity.size() );
   for( unsigned i = 0; i < per_arity.size(); ++i )
   {
      if( !per_arity[i].empty() ) // there is at least one primitive with such arity
      {
         m_primitives.insert( m_primitives.end(), per_arity[i].begin(), per_arity[i].end() );
         m_primitives_boundaries[i].first = m_primitives.size() - per_arity[i].size();
         m_primitives_boundaries[i].second = m_primitives.size() - 1;
      }
      else
         m_primitives_boundaries[i] = m_primitives_boundaries[i - 1];
   }

   // Fill m_max_arity
   m_max_arity = per_arity.size() - 1;

   // ----------------------------------------------
   std::cout << std::endl;
   for( int i = 0; i < m_primitives.size(); ++i )
   {
      std::cout << "[" << i << "] Arity: " << ARITY( m_primitives[i] ) << " Index: " << INDEX( m_primitives[i] ) << std::endl;
   }

   std::cout << std::endl;
   for( int i = 0; i < m_primitives_boundaries.size(); ++i )
   {
      std::cout << "Boundaries " << i << " : " << m_primitives_boundaries[i].first << "," << m_primitives_boundaries[i].second << std::endl;
   }
}

// -----------------------------------------------------------------------------
cl_uint Primitives::RandomGene( unsigned min, unsigned max )
{
   // Truncate to m_max_arity if necessary
   max = std::min( m_max_arity, max );

   cl_uint gene = m_primitives[Random::Int( m_primitives_boundaries[min].first, 
                                            m_primitives_boundaries[max].second )];

   // Handle the ephemeral constants because they need to get a random value on-the-fly
   if( INDEX( gene ) == GPT_EPHEMERAL )
      gene = PackAlelo( 0, GPT_EPHEMERAL, (float) Random::Real( 0.0, SCALE_FACTOR ) );

   // The returned cl_uint is a packed gene
   return gene;
}

// -----------------------------------------------------------------------------
