#include <mpi.h>
#include "header/MPIHandler.h"
#include "header/SIRModel.h"
#include "header/CSVParser.h"
#include "header/GridSimulation.h"
#include <iostream>
#include <unordered_map>
#include <map>
#include <list>
#include <vector>
#include <string>

int main(int argc, char *argv[]) {
    MPIHandler mpi(argc, argv);

    if (argc < 1) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create SIR model
    SIRModel model(0.3, 0.1, 0.2, 100);

    // Load data and distribute it
    std::vector<std::vector<double>> fullData;
    if (mpi.getRank() == 0) {
        fullData = CSVParser::loadUSStateData("./data/sorted_initial_conditions.csv");
    }

    std::vector<SIRCell> localGrid = mpi.distributeData(fullData, [](const std::vector<double>& rowData) {
        return CSVParser::mapToSIR(rowData); // Call static method from CSVParser
    });

    // Create and configure simulation
    GridSimulation simulation(model, mpi.getRank(), mpi.getSize());
    simulation.initialize(localGrid, mpi.getSize());

    // Run the simulation
    std::vector<std::vector<double>> localResults = simulation.runSimulation();

    // Gather and write results
    std::vector<double> globalFlatResults = mpi.gatherResults(localResults);
    if (mpi.getRank() == 0) {
        mpi.writeResults(globalFlatResults, localResults.size());
    }

    return 0;
}
