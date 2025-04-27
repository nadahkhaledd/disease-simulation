#include "../header/CSVParser.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include <unordered_map>
#include <map>
#include <algorithm>

std::string CSVParser::trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

std::vector<std::vector<double>> CSVParser::loadUSStateData(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    std::string line;
    bool headerFound = false;
    int lineCount = 0;
    
    // Each row will be stored as a vector of double
    std::vector<std::vector<double>> data;
    
    // Process lines until we find valid data
    while (std::getline(infile, line)) {
        lineCount++;
        
        // Skip empty lines
        if (line.empty()) continue;
        
        // Check if this line appears to be the header
        if (line.find("Province_State") != std::string::npos) {
            std::cout << "Found header at line " << lineCount << ": " << line << std::endl;
            headerFound = true;
            continue;
        }
        
        // Skip any lines before the header
        if (!headerFound) {
            std::cout << "Skipping pre-header line: " << line << std::endl;
            continue;
        }
        
        // Process data line
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        // Split line on commas
        while (std::getline(ss, token, ',')) {
            tokens.push_back(trim(token));
        }
        
        // Verify we have enough columns
        if (tokens.size() < 9) {
            std::cerr << "Line " << lineCount << " has fewer than 9 columns: " << line << std::endl;
            continue;
        }
        
        try {
            // Convert values to appropriate types
            int population = std::stoi(tokens[1]); // Population as integer
            double lat = std::stod(tokens[3]);
            double lon = std::stod(tokens[4]);
            double confirmed = std::stod(tokens[5]);
            double deaths = std::stod(tokens[6]);
            double recovered = std::stod(tokens[7]);
            double active = std::stod(tokens[8]);
            
            // Store values (convert population to double for consistency in the data structure)
            data.push_back({static_cast<double>(population), lat, lon, confirmed, deaths, recovered, active});
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid value at line " << lineCount << ": " << line << "\nError: " << e.what() << std::endl;
            continue;
        } catch (const std::out_of_range& e) {
            std::cerr << "Value out of range at line " << lineCount << ": " << line << "\nError: " << e.what() << std::endl;
            continue;
        }
    }
    
    std::cout << "Successfully parsed " << data.size() << " data rows from CSV." << std::endl;
    return data;
}

SIRCell CSVParser::mapToSIR(const std::vector<double>& rowData) {
    // Adjusted to match the correct column structure:
    // Province_State, Population, Date, Lat, Long, Confirmed, Deaths, Recovered, Active
    double population = rowData[0]; // Population
    double confirmed = rowData[3]; // Confirmed cases
    double deaths = rowData[4];    // Deaths
    double recovered = rowData[5]; // Recovered
    double active = rowData[6];    // Active cases

    // Ensure population is valid
    if (population <= 0) {
        std::cerr << "Error: Invalid population value in input data.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Calculate S, I, R based on the population
    double S = (population - confirmed) / population;
    double I = active / population;
    double R = (recovered + deaths) / population;

    // Ensure S, I, R are within valid bounds
    if (S < 0) S = 0;
    if (I < 0) I = 0;
    if (R < 0) R = 0;

    return SIRCell(S, I, R);
}