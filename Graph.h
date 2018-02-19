#pragma once

#include "Field.h"

#include <matplotlib-cpp.h>

template<int N1, int N2, int N3>
inline void HeatPlot1D(const Nodal1D<N1, N2, N3> &U, std::string filename)
{
    if_root
    {
        matplotlibcpp::figure();

        int cols = N3;
        int rows = N3;

        std::vector<stratifloat> imdata(rows*cols);

        for (int col=0; col<cols; col++)
        {
            for (int row=0; row<rows; row++)
            {
                imdata[row*cols + col] = U.Get()(row);
            }
        }

        matplotlibcpp::imshow(imdata, rows, cols);

        matplotlibcpp::save(filename);
        matplotlibcpp::close();
    } endif
}

#ifdef DEBUG_PLOT

template<int N1, int N2, int N3>
inline void HeatPlot(const NodalField<N1, N2, N3> &U, stratifloat L1, stratifloat L3, int j2, std::string filename)
{
    if_root
    {
        matplotlibcpp::figure();

        std::vector<stratifloat> imdata(N1*N3);

        for (int n1=0; n1<N1; n1++)
        {
            for (int n3=0; n3<N3; n3++)
            {
                imdata[n3*N1 + n1] = U(n1,j2,n3);
            }
        }

        matplotlibcpp::imshow(imdata, N3, N1);

        matplotlibcpp::save(filename);
        matplotlibcpp::close();
    } endif
}

template<int N1, int N2, int N3>
inline void HeatPlot(const ModalField<N1, N2, N3> &u, stratifloat L1, stratifloat L3, int j2, std::string filename)
{
    NodalField<N1, N2, N3> U(u.BC());

    u.ToNodal(U);

    HeatPlot(U, L1, L3, j2, filename);
}

#else


#endif