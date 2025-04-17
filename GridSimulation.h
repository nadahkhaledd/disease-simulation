#ifndef GRIDSIMULATION_H
#define GRIDSIMULATION_H

#include <vector>
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

    
    std::vector<std::vector<double>> runSimulation();
};

#endif // GRIDSIMULATION_H