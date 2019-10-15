using NeuralNetwork.Tools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Classes
{
    public class Neuron
    {
        public int Index { get; set; } // For multi-threading purposes
        public double Bias { get; set; }
        public double BiasDelta { get; set; }
        public double Gradient { get; set; }
        public double Value { get; set; }
        public IList<Synapse> Inputs { get; set; }
        public IList<Synapse> Outputs { get; set; }

        public Neuron()
        {
            Inputs = new List<Synapse>();
            Outputs = new List<Synapse>();
            Bias = Network.GetRandom();
        }

        public Neuron(IEnumerable<Neuron> inputNeurons) : this()
        {
            foreach (var inputNeuron in inputNeurons)
            {
                var synapse = new Synapse(inputNeuron, this);
                inputNeuron.Outputs.Add(synapse);
                Inputs.Add(synapse);
            }
        }

        public virtual double CalculateValue(bool dropOut)
        {
            //if(dropOut && Extensions.rng.NextDouble() < 0.2d)
            //{
            //    wasDropped = true;
            //    return 0.0d;
            //}
            //wasDropped = false;

            //Value = (Inputs.Sum(a => a.Weight * a.Input.Value) + Bias).ReLU((double)Inputs.Count());

            Value = (Inputs.Sum(a => a.Weight * a.Input.Value) + Bias).Sigmoid(Inputs.Count()); 

            //Value = Inputs.Select(s => s.Weight * s.Input.Value).ToList().SoftArgMax();

            return Value;
        }

        public double CalculateError(double target)
        {
            return target - Value;
        }

        public double CalculateGradient(double? target = null)
        {
            if (target == null)
                return Gradient = Outputs.Sum(a => a.Output.Gradient * a.Weight) * Value.Derivative();

            return Gradient = CalculateError(target.Value) * Value.Derivative();
        }

        public void UpdateWeights(double learnRate, double momentum)
        {
            var prevDelta = BiasDelta;
            BiasDelta = learnRate * Gradient;
            Bias += BiasDelta + momentum * prevDelta;

            foreach (var synapse in Inputs)
            {
                prevDelta = synapse.WeightDelta;
                synapse.WeightDelta = learnRate * Gradient * synapse.Input.Value;
                synapse.Weight += synapse.WeightDelta + momentum * prevDelta;
            }
        }
    }
}