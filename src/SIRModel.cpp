#include "../header/SIRModel.h"

SIRModel::SIRModel(double b, double g, double timeStep, int steps)
    : beta(b), gammaRate(g), dt(timeStep), numSteps(steps) {}

double SIRModel::getBeta() const { 
    return beta; 
}

double SIRModel::getGamma() const { 
    return gammaRate; 
}

double SIRModel::getDt() const { 
    return dt; 
}

int SIRModel::getNumSteps() const { 
    return numSteps; 
}

SIRCell SIRModel::rk4Step(const SIRCell &current) const {
    double S = current.getS();
    double I = current.getI();
    double R = current.getR();

    // Ensure S, I, R are within valid bounds
    if (S < 0) S = 0;
    if (I < 0) I = 0;
    if (R < 0) R = 0;

    auto fS = [&](double s, double i) -> double {
        return -beta * s * i;
    };
    auto fI = [&](double s, double i) -> double {
        return beta * s * i - gammaRate * i;
    };
    auto fR = [&](double i) -> double {
        return gammaRate * i;
    };

    // RK4 steps
    double k1_S = dt * fS(S, I);
    double k1_I = dt * fI(S, I);
    double k1_R = dt * fR(I);

    double k2_S = dt * fS(S + 0.5 * k1_S, I + 0.5 * k1_I);
    double k2_I = dt * fI(S + 0.5 * k1_S, I + 0.5 * k1_I);
    double k2_R = dt * fR(I + 0.5 * k1_I);

    double k3_S = dt * fS(S + 0.5 * k2_S, I + 0.5 * k2_I);
    double k3_I = dt * fI(S + 0.5 * k2_S, I + 0.5 * k2_I);
    double k3_R = dt * fR(I + 0.5 * k2_I);

    double k4_S = dt * fS(S + k3_S, I + k3_I);
    double k4_I = dt * fI(S + k3_S, I + k3_I);
    double k4_R = dt * fR(I + k3_I);

    double newS = S + (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6.0;
    double newI = I + (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6.0;
    double newR = R + (k1_R + 2 * k2_R + 2 * k3_R + k4_R) / 6.0;

    // Ensure S, I, R remain within [0, 1]
    if (newS < 0) newS = 0;
    if (newI < 0) newI = 0;
    if (newR < 0) newR = 0;

    return SIRCell(newS, newI, newR);
}

SIRCell SIRModel::rk4StepWithNeighbors(const SIRCell& current, const std::vector<SIRCell>& neighbors) const {
    double S = current.getS();
    double I = current.getI();
    double R = current.getR();

    // 💉 Compute average neighbor infection level (Ij from neighbors)
    double totalI = 0.0;
    for (const auto& neighbor : neighbors) {
        totalI += neighbor.getI();
    }

    double avgI = neighbors.empty() ? 0.0 : totalI / neighbors.size();

    // Modified equations using avgI from neighbors
    auto fS = [&](double s) -> double {
        return -beta * s * avgI;
    };
    auto fI = [&](double s, double i) -> double {
        return beta * s * avgI - gammaRate * i;
    };
    auto fR = [&](double i) -> double {
        return gammaRate * i;
    };

    // RK4 method
    double k1_S = dt * fS(S);
    double k1_I = dt * fI(S, I);
    double k1_R = dt * fR(I);

    double k2_S = dt * fS(S + 0.5 * k1_S);
    double k2_I = dt * fI(S + 0.5 * k1_S, I + 0.5 * k1_I);
    double k2_R = dt * fR(I + 0.5 * k1_I);

    double k3_S = dt * fS(S + 0.5 * k2_S);
    double k3_I = dt * fI(S + 0.5 * k2_S, I + 0.5 * k2_I);
    double k3_R = dt * fR(I + 0.5 * k2_I);

    double k4_S = dt * fS(S + k3_S);
    double k4_I = dt * fI(S + k3_S, I + k3_I);
    double k4_R = dt * fR(I + k3_I);

    double newS = S + (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6.0;
    double newI = I + (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6.0;
    double newR = R + (k1_R + 2 * k2_R + 2 * k3_R + k4_R) / 6.0;

    return SIRCell(newS, newI, newR);
}