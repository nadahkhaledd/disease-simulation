#ifndef MPIHANDLER_H
#define MPIHANDLER_H

#include <vector>
#include "SIRCell.h"
#include <functional>

class MPIHandler {
private:
    int rank, size;
    
public:
    MPIHandler(int argc, char *argv[]);
    ~MPIHandler();
    
    int getRank() const;
    int getSize() const;
    
    // Distribute data among processes
    std::vector<SIRCell> distributeData(const std::vector<std::vector<double>>& fullData,
                                        const std::function<SIRCell(const std::vector<double>&)>& mapFunction);
    
    // Gather results from all processes
    std::vector<double> gatherResults(const std::vector<std::vector<double>>& localResults);
    
    // Write results to file
    void writeResults(const std::vector<double>& globalFlat, int steps);
};

#endif // MPIHANDLER_H