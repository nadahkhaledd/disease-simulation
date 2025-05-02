#ifndef MPIHANDLER_H
#define MPIHANDLER_H

#include <vector>         // For std::vector
#include <map>            // For std::map
#include <list>           // For std::list
#include <string>         // For std::string (often useful)
#include <unordered_map>  // For std::unordered_map

// Forward declare classes used in function signatures if their full definition
// isn't required within this header file itself. This can help reduce compile times.
class SIRCell;
#include <vector>
#include "SIRCell.h"
#include <functional>

class MPIHandler {
public:
    // Constructor: Initializes MPI environment.
    MPIHandler(int argc, char *argv[]);

    // Destructor: Finalizes MPI environment.
    ~MPIHandler();

    // --- Getters ---
    // Returns the MPI rank (ID) of the current process.
    int getRank() const;
    int getSize() const;

    // --- MPI Communication / Data Handling Methods ---

    // Distributes initial cell data using a simple row-based division.
    // (May be less relevant in the block-based workflow).
    std::vector<SIRCell> distributeData(const std::vector<std::vector<double>>& fullData);

    // Gathers simulation results (e.g., [time, avgS, avgI, avgR]) from all processes
    // onto Rank 0. Uses MPI_Gatherv for potentially variable data sizes per rank.

    // Distribute data among processes
    std::vector<SIRCell> distributeData(const std::vector<std::vector<double>>& fullData,
                                        const std::function<SIRCell(const std::vector<double>&)>& mapFunction);

    // Gather results from all processes
    std::vector<double> gatherResults(const std::vector<std::vector<double>>& localResults);

    // Writes the globally gathered simulation results to a CSV file.
    // Typically called only by Rank 0.
    void writeResults(const std::vector<double>& globalFlat, int steps_hint); // steps_hint may not be strictly needed

    // Distributes the block structure (map of BlockID -> list<CellID>) from Rank 0
    // to all other processes. Sends only the structure, not the cell data itself.
    std::map<int, std::list<int>> distributeBlocks(
        const std::map<int, std::list<int>>& allBlocks); // Only needs allBlocks on Rank 0

    // Handles the process where each rank requests the specific initial condition data rows
    // it needs based on its assigned blocks, and Rank 0 sends back the requested data.
    // Returns a map of CellID -> data_row (vector<double>) for the local process.
    std::map<int, std::vector<double>> getDataForLocalBlocks(
        const std::map<int, std::list<int>>& localBlocks,    // Blocks assigned to this rank
        const std::vector<std::vector<double>>& fullData); // Full dataset (only used by Rank 0)

    // Broadcasts the block adjacency map (BlockID -> vector<neighbor BlockID>)
    // from Rank 0 to all other processes.
    std::unordered_map<int, std::vector<int>> broadcastBlockNeighborMap(
        const std::unordered_map<int, std::vector<int>>& mapToSend); // Map only needed on Rank 0 initially

private:
    // --- Private Member Variables ---
    int rank; // MPI rank of the current process.
    int size; // Total number of MPI processes.
};

#endif // MPI_HANDLER_H