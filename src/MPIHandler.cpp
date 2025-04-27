#include "../header/MPIHandler.h"
#include "../header/CSVParser.h"
#include <mpi.h>
#include <iostream>
#include <fstream>

MPIHandler::MPIHandler(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

MPIHandler::~MPIHandler() {
    MPI_Finalize();
}

int MPIHandler::getRank() const { 
    return rank; 
}

int MPIHandler::getSize() const { 
    return size; 
}

std::vector<SIRCell> MPIHandler::distributeData(const std::vector<std::vector<double>>& fullData,
                                                const std::function<SIRCell(const std::vector<double>&)>& mapFunction) {
    std::vector<SIRCell> localGrid;
    
    // Debug: Print data size on each rank
    if (rank == 0) {
        std::cout << "Rank 0 has full data with " << fullData.size() << " rows" << std::endl;
    } else {
        std::cout << "Rank " << rank << " initially has 0 rows (no data)" << std::endl;
    }
    
    int totalRows = 0;
    if (rank == 0) {
        totalRows = static_cast<int>(fullData.size());
    }
    MPI_Bcast(&totalRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Debug: Everyone knows total rows after broadcast
    std::cout << "Rank " << rank << " knows there are " << totalRows << " total rows after broadcast" << std::endl;
    
    int rowsPerProc = totalRows / size;
    int extra = totalRows % size;
    int startIndex, localRows;
    
    if (rank < extra) {
        localRows = rowsPerProc + 1;
        startIndex = rank * localRows;
    } else {
        localRows = rowsPerProc;
        startIndex = rank * localRows + extra;
    }
    
    // Debug: Show assigned ranges
    std::cout << "Rank " << rank << " is assigned rows " << startIndex << " to " 
              << (startIndex + localRows - 1) << " (" << localRows << " rows)" << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Process 0 handles its own portion
        for (int i = startIndex; i < startIndex + localRows && i < static_cast<int>(fullData.size()); i++) {
            localGrid.push_back(mapFunction(fullData[i]));
        }
        
        // Debug: Confirm rank 0's own data assignment
        std::cout << "Rank 0 kept " << localGrid.size() << " rows for itself" << std::endl;
        
        // Send portions to other processes
        for (int proc = 1; proc < size; proc++) {
            int procRows = (proc < extra) ? rowsPerProc + 1 : rowsPerProc;
            int procStart = (proc < extra) ? proc * (rowsPerProc + 1) : proc * rowsPerProc + extra;
            std::vector<double> sendBuffer;
            
            for (int i = procStart; i < procStart + procRows && i < static_cast<int>(fullData.size()); i++) {
                SIRCell cell = mapFunction(fullData[i]);
                sendBuffer.push_back(cell.getS());
                sendBuffer.push_back(cell.getI());
                sendBuffer.push_back(cell.getR());
            }
            
            // Debug: Show what's being sent
            std::cout << "Rank 0 sending " << sendBuffer.size()/3 << " rows to rank " << proc << std::endl;
            
            // Send the data
            if (!sendBuffer.empty()) {
                MPI_Send(sendBuffer.data(), sendBuffer.size(), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            } else {
                // Send empty signal
                double dummy = -1.0;
                MPI_Send(&dummy, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
                std::cout << "Rank 0 sent empty data signal to rank " << proc << std::endl;
            }
        }
    } else {
        // Other processes receive their portion
        std::vector<double> recvBuffer(localRows * 3);
        MPI_Status status;
        MPI_Recv(recvBuffer.data(), localRows * 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        
        // Debug: Check how much data was actually received
        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        std::cout << "Rank " << rank << " received " << count << " doubles (" << count/3 << " rows)" << std::endl;
        
        // If we received only one element with value -1, it's the empty signal
        if (count == 1 && recvBuffer[0] == -1.0) {
            std::cout << "Rank " << rank << " received empty data signal" << std::endl;
        } else {
            // Process the received data
            for (int i = 0; i < count/3; i++) {
                SIRCell cell(recvBuffer[3*i], recvBuffer[3*i+1], recvBuffer[3*i+2]);
                localGrid.push_back(cell);
            }
        }
    }
    
    // Debug: Final summary after all data is distributed
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << rank << " has " << localGrid.size() 
              << " rows in its final local grid" << std::endl;
    
    // Verify total distributed rows
    int localSize = static_cast<int>(localGrid.size());
    int totalSize;
    MPI_Reduce(&localSize, &totalSize, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Total distributed rows: " << totalSize 
                  << " (original size: " << fullData.size() << ")" << std::endl;
    }
    
    return localGrid;
}

std::vector<double> MPIHandler::gatherResults(const std::vector<std::vector<double>>& localResults) {
    int steps = static_cast<int>(localResults.size());
    std::vector<double> localFlat;

    // Flatten local results
    for (const auto& row : localResults) {
        localFlat.insert(localFlat.end(), row.begin(), row.end());
    }

    // Prepare global buffer on rank 0
    std::vector<double> globalFlat;
    if (rank == 0) {
        globalFlat.resize(steps * 4 * size); // 4 columns: [time, S, I, R]
    }

    // Gather results from all processes
    MPI_Gather(localFlat.data(), steps * 4, MPI_DOUBLE,
               globalFlat.data(), steps * 4, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // Normalize results on rank 0
    if (rank == 0) {
        for (size_t i = 0; i < globalFlat.size(); i += 4) {
            double S = globalFlat[i + 1];
            double I = globalFlat[i + 2];
            double R = globalFlat[i + 3];
            double sum = S + I + R;

            if (sum > 0) {
                globalFlat[i + 1] = std::max(0.0, std::min(1.0, S / sum));
                globalFlat[i + 2] = std::max(0.0, std::min(1.0, I / sum));
                globalFlat[i + 3] = std::max(0.0, std::min(1.0, R / sum));
            } else {
                globalFlat[i + 1] = 1.0;
                globalFlat[i + 2] = 0.0;
                globalFlat[i + 3] = 0.0;
            }
        }
    }

    return globalFlat;
}

void MPIHandler::writeResults(const std::vector<double>& globalFlat, int steps) {
    if (rank == 0) {
        std::ofstream outfile("simulation_results.csv");
        outfile << "Process,Time,S,I,R\n";
        
        for (int proc = 0; proc < size; proc++) {
            for (int i = 0; i < steps; i++) {
                int index = (proc * steps + i) * 4;
                outfile << proc << "," 
                        << globalFlat[index] << ","
                        << globalFlat[index+1] << ","
                        << globalFlat[index+2] << ","
                        << globalFlat[index+3] << "\n";
            }
        }
        
        outfile.close();
        std::cout << "Results written to simulation_results.csv" << std::endl;
    }
}