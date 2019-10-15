using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Classes
{
    public class Synapse
    {
        public Neuron Input { get; set; }
        public Neuron Output { get; set; }
        public double Weight { get; set; }
        public double WeightDelta { get; set; }

        // Empty constructor for serialization
        public Synapse()
        {

        }

        public Synapse(Neuron input, Neuron output)
        {
            Input = input;
            Output = output;
            Weight = Network.GetRandom();
        }
    }
}