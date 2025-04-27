#ifndef GRIDSIMULATION_H
#define GRIDSIMULATION_H

#include <vector>
#include <unordered_map>
#include <map>
#include <list>
#include <string>
#include "SIRCell.h"
#include "SIRModel.h"
#include <unordered_map>

class GridSimulation {
private:
    std::vector<SIRCell> grid;
    SIRModel model;
    int rank, size;
    std::unordered_map<int, std::vector<int>> neighborMap;

public:
    GridSimulation(const SIRModel& m, int mpiRank, int mpiSize);
    
    void setGrid(const std::vector<SIRCell>& initialGrid);
    
    std::vector<SIRCell>& getGrid();
    
    int getLocalSize() const;
    
    void updateGrid();
    void updateGridNew();
    void setNeighborMap(const std::unordered_map<int, std::vector<int>>& map);

    void initialize(const std::vector<SIRCell>& localGrid, int numProcesses);
    static std::unordered_map<int, std::vector<int>> build2DGridNeighborMap(int rows, int cols);
    static std::pair<int, int> calculateGridDimensions(int totalCells, int numBlocks);
    
    std::vector<std::vector<double>> runSimulation();

    static std::map<std::string, int> createCellsMap();
    static std::map<int, std::list<int>> divideIntoBlocks(
        const std::map<std::string, int>& cells, int blockSize);
    static std::map<int, std::list<int>> divideIntoOptimalBlocks(
        const std::map<std::string, int>& cells, int numProcesses);
};

#endif // GRIDSIMULATION_H