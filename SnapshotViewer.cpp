#include "Stratiflow.h"
#include "Graph.h"
#include "OSUtils.h"

#include <string>
#include <fstream>
#include <dirent.h>
#include <iostream>
#include <map>


int main(int argc, char *argv[])
{
    NField loaded(BoundaryCondition::Decaying);

    LoadVariable(argv[1], loaded, 2);
    MField loadedModal(BoundaryCondition::Decaying);

    loaded.ToModal(loadedModal);

    HeatPlot(loadedModal, L1, L3, 0, "output.png");

    return 0;
}