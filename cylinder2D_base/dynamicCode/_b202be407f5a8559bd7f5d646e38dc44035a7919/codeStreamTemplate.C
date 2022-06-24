/*---------------------------------------------------------------------------*  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) YEAR OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Description
    Template for use with codeStream.

\*---------------------------------------------------------------------------*/

#include "dictionary.H"
#include "Ostream.H"
#include "Pstream.H"
#include "unitConversion.H"

//{{{ begin codeInclude
#line 35 "/home/jan/POD-ROM-fluid-flows/cylinder2D_base/system/blockMeshDict/#codeStream"
#include "pointField.H"
//}}} end codeInclude

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    void codeStream_b202be407f5a8559bd7f5d646e38dc44035a7919
    (
        Ostream& os,
        const dictionary& dict
    )
    {
//{{{ begin code
        #line 40 "/home/jan/POD-ROM-fluid-flows/cylinder2D_base/system/blockMeshDict/#codeStream"
pointField points({
            /* 0*/ {0, 0, 0},
            /* 1*/ {2.00000000e-01 * 2, 0, 0},
            /* 2*/ {2.20000000e+00, 0, 0},
            /* 3*/ {2.20000000e+00, 4.10000000e-01, 0},
            /* 4*/ {2.00000000e-01 * 2, 4.10000000e-01, 0},
            /* 5*/ {0, 4.10000000e-01, 0},
            /* 6*/ {2.00000000e-01 - 3.53553380e-02, 2.00000000e-01 - 3.53553380e-02, 0},
            /* 7*/ {2.00000000e-01 + 3.53553380e-02, 2.00000000e-01 - 3.53553380e-02, 0},
            /* 8*/ {2.00000000e-01 - 3.53553380e-02, 2.00000000e-01 + 3.53553380e-02, 0},
            /* 9*/ {2.00000000e-01 + 3.53553380e-02, 2.00000000e-01 + 3.53553380e-02, 0}
        });

        // Duplicate z points for thickness
        const label sz = points.size();
        points.resize(2*sz);
        for (label i = 0; i < sz; ++i)
        {
            const point& pt = points[i];
            points[i + sz] = point(pt.x(), pt.y(), 1.00000000e-02);
        }

        os  << points;
//}}} end code
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

