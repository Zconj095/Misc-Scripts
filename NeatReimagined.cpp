#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <map>

struct ConnectionGene {
  int from; 
  int to;
  double weight;
};

class Genome {
public:
  std::vector<ConnectionGene> connections;
  
  void addConnection(int from, int to, double weight) {
    ConnectionGene cg;
    cg.from = from;
    cg.to = to; 
    cg.weight = weight;
    connections.push_back(cg);  
  }
  
  void mutate() {
    // Add new random connection
    int randomFrom = rand() % numNodes;
    int randomTo = rand() % numNodes;
    double randomWeight = ((double) rand()) / (RAND_MAX);
    addConnection(randomFrom, randomTo, randomWeight);
    
    // Randomly alter weights
    for (auto& cg : connections) {
      if (rand() % 2 == 0) { 
        cg.weight += 0.1 * (rand() % 2 == 0 ? -1 : 1);  
      }
    }
  }
  
private:
  int numNodes; 
};

class Population {
public:
  std::vector<Genome> genomes; 
  
  Genome breed(Genome g1, Genome g2) {
    Genome child;
    
    // Inherit connections from parents with crossover
    std::random_shuffle(g1.connections.begin(), g1.connections.end());
    std::random_shuffle(g2.connections.begin(), g2.connections.end());
    
    int split = rand() % g1.connections.size();
    for (int i = 0; i < split; i++) {
      child.addConnection(g1.connections[i].from, 
                         g1.connections[i].to,
                         g1.connections[i].weight); 
    }
    
    for (int i = split; i < g2.connections.size(); i++) {
      child.addConnection(g2.connections[i].from, 
                         g2.connections[i].to,
                         g2.connections[i].weight);
    }
    
    // Mutate child
    child.mutate();
    
    return child;
  }
  
  void evolve() {
    std::sort(genomes.begin(), genomes.end(), 
              [](const Genome& a, const Genome& b) {
                return a.fitness > b.fitness;  
              });
              
    std::vector<Genome> newGenomes;
    
    // Elite selection
    newGenomes.push_back(genomes[0]);
    
    // Tournament selection 
    int tournamentSize = 5;
    for (int i = 0; i < genomes.size(); i++) {
      std::vector<Genome> tournament;
      for (int j = 0; j < tournamentSize; j++) {
        int randomIndex = rand() % genomes.size();
        tournament.push_back(genomes[randomIndex]);  
      }
      
      std::sort(tournament.begin(), tournament.end(), 
                [](const Genome& a, const Genome& b) {
                  return a.fitness > b.fitness;
                });
                
      Genome g1 = tournament[0];
      Genome g2 = tournament[1];
      
      Genome child = breed(g1, g2);
      newGenomes.push_back(child);
    }
    
    // Replace old generation with new generation      
    genomes = newGenomes;
  }
  
};

int main() {
  
  Population pop;
  // Initialize population
  
  for (int gen = 0; gen < 100; gen++) {
    // Evaluate fitness
    
    pop.evolve();   
  }

  return 0;  
}
