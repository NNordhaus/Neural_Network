using NeuralNetwork.Tools;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Classes
{
    public class Network
    {
        public double LearnRate { get; set; }
        public double Momentum { get; set; }
        public List<Neuron> InputLayer { get; set; }
        public List<List<Neuron>> HiddenLayers { get; set; }
        public List<Neuron> OutputLayer { get; set; }
        [JsonIgnore]
        public int NeuronCount{ get { return InputLayer.Count + HiddenLayers.Sum(l=>l.Count) + OutputLayer.Count; }}
        [JsonIgnore]
        public int SynapseCount
        {
            get
            {
                return InputLayer.Sum(n => n.Outputs.Count())
                + HiddenLayers.Sum(h => h.Sum(n => n.Outputs.Count()));
            }
        }
        [JsonIgnore]
        public string Config
        {
            get
            {
                return InputLayer.Count + "-" + string.Join("-", HiddenLayers.Select(l => l.Count.ToString()).ToArray())
                    + "-" + OutputLayer.Count + " " + NeuronCount + " total neurons, " + SynapseCount.ToString("#,#") + " total synapses, " + Environment.NewLine
                    + LearnRate + " learn rate, " + Momentum + " momentum.";
            }
        }

        public static readonly Random Random = new Random(DateTime.Now.Millisecond * 7321);

        public Network()
        {
            //LearnRate = 0;
            //Momentum = 0;
            //InputLayer = new List<Neuron>();
            //HiddenLayers = new List<List<Neuron>>();
            //OutputLayer = new List<Neuron>();
        }

        public Network(int inputSize, int[] hiddenSizes, int outputSize, double? learnRate = null, double? momentum = null)
        {
            LearnRate = learnRate ?? .6;
            Momentum = momentum ?? .1;
            InputLayer = new List<Neuron>();
            HiddenLayers = new List<List<Neuron>>();
            OutputLayer = new List<Neuron>();

            for (var i = 0; i < inputSize; i++)
            {
                InputLayer.Add(new Neuron() { Index = i });
            }

            var firstHiddenLayer = new List<Neuron>();
            for (var i = 0; i < hiddenSizes[0]; i++)
                firstHiddenLayer.Add(new Neuron(InputLayer));

            HiddenLayers.Add(firstHiddenLayer);

            for (var i = 1; i < hiddenSizes.Length; i++)
            {
                var hiddenLayer = new List<Neuron>();
                for (var j = 0; j < hiddenSizes[i]; j++)
                    hiddenLayer.Add(new Neuron(HiddenLayers[i - 1]));
                HiddenLayers.Add(hiddenLayer);
            }

            for (var i = 0; i < outputSize; i++)
                OutputLayer.Add(new Neuron(HiddenLayers.Last()));
        }

        #region -- Training --
        public void Train(IList<DataSet> dataSets, IList<DataSet> testSet, int numEpochs)
        {
            for (var i = 0; i < numEpochs; i++)
            {
                dataSets.Shuffle();
                var errors = new List<double>();
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(true, dataSet.Values);
                    BackPropagate(dataSet.Targets);
                    var error = CalculateError(dataSet.Targets);
                    errors.Add(error);

                    Console.SetCursorPosition(0, Console.CursorTop);
                    Console.Write(errors.Count().ToString("#,###,###") + "\t " + errors.Average().ToString("0.000000") + "      ");
                }
                Console.SetCursorPosition(0, Console.CursorTop-1);

                var avg = errors.Average();
                //var max = errors.Max();
                //var min = errors.Min();
                var med = errors.Median();
                //Console.WriteLine(Environment.NewLine +
                //    i + " mean: " + avg.ToString("0.000000") + "  median: " + med.ToString("0.000000") + "  min: " + min.ToString("0.000000") + "  max: " + max.ToString("0.000000")
                //    + "  " + DateTime.Now.ToString("H:mm:ss"));
                Console.WriteLine(Environment.NewLine +
                    i + " mean: " + avg.ToString("0.000000") + "  median: " + med.ToString("0.000000")
                    + "  " + DateTime.Now.ToString("H:mm:ss"));

                Console.WriteLine("Failure Rate: " + Fitness_FailureRate(testSet).ToString("0.0000%") + "\t   ");

                // Check for synapses with near zero weights
                int deadSynapseCount = 0;
                /*
                var allSynapses = InputLayer.SelectMany(n => n.Outputs)
                    .Union(HiddenLayers.SelectMany(h => h.SelectMany(n => n.Outputs)))
                    .ToArray();

                foreach (var s in allSynapses)
                {
                    if(Math.Abs(s.Weight) <= 0.0001d)
                    {
                        s.Input.Outputs.Remove(s);
                        s.Output.Inputs.Remove(s);
                        deadSynapseCount++;
                    }
                }

                Console.WriteLine(deadSynapseCount + " near-zero synapses removed");
                */
            }
        }

        public void Train(List<DataSet> dataSets, double minimumError)
        {
            var error = 1.0;
            var numEpochs = 0;

            while (error > minimumError && numEpochs < int.MaxValue)
            {
                dataSets.Shuffle();
                var errors = new List<double>();
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(true, dataSet.Values);
                    BackPropagate(dataSet.Targets);
                    errors.Add(CalculateError(dataSet.Targets));
                }
                error = errors.Average();
                numEpochs++;
                //Console.WriteLine(Fitness(dataSets));
                Console.WriteLine(error);
                this.LearnRate -= 0.00001;
            }
        }

        private void ForwardPropagate(bool dropOut, params double[] inputs)
        {
            bool parallel = false;

            if (parallel)
            {
                Parallel.ForEach(InputLayer, (a) =>
                {
                    a.Value = inputs[a.Index];
                });
            }
            else
            {
                InputLayer.ForEach(a => a.Value = inputs[a.Index]);
            }

            if (parallel)
            {
                HiddenLayers.ForEach(a =>
                    Parallel.ForEach(a, (b) =>
                    {
                        b.CalculateValue(dropOut);
                    })
                );
            }
            else
            {
                HiddenLayers.ForEach(a => a.ForEach(b => b.CalculateValue(dropOut)));
            }

            OutputLayer.ForEach(a => a.CalculateValue(false));
        }

        private void BackPropagate(params double[] targets)
        {
            var i = 0;
            OutputLayer.ForEach(a => a.CalculateGradient(targets[i++]));
            HiddenLayers.Reverse();
            //HiddenLayers.ForEach(a => a.ForEach(b => b.CalculateGradient()));
            HiddenLayers.ForEach(a =>
                Parallel.ForEach(a, (b) =>
                {
                    b.CalculateGradient();
                })
            );
            //HiddenLayers.ForEach(a => a.ForEach(b => b.UpdateWeights(LearnRate, Momentum)));
            HiddenLayers.ForEach(a =>
                Parallel.ForEach(a, (b) =>
                {
                    b.UpdateWeights(LearnRate, Momentum);
                })
            );
            HiddenLayers.Reverse();
            //OutputLayer.ForEach(a => a.UpdateWeights(LearnRate, Momentum));
            Parallel.ForEach(OutputLayer, (on) =>
            {
                on.UpdateWeights(LearnRate, Momentum);
            });
        }

        public double[] Compute(params double[] inputs)
        {
            ForwardPropagate(true, inputs);
            return OutputLayer.Select(a => a.Value).ToArray();
        }

        private double CalculateError(params double[] targets)
        {
            //var i = 0;
            //return OutputLayer.Sum(a => Math.Abs(a.CalculateError(targets[i++])));

            double sum = 0.0;
            for(int i = 0; i < targets.Length; i++)
            {
                sum += Math.Abs(OutputLayer[i].CalculateError(targets[i]));
            }
            return sum;
        }
        #endregion

        #region -- Grading --
        public double Fitness(List<DataSet> testDataSets)
        {
            //var grade = 1.0;
            var errors = new List<double>();
            foreach (var dataSet in testDataSets)
            {
                ForwardPropagate(false, dataSet.Values);
                errors.Add(Math.Abs(CalculateError(dataSet.Targets)));
            }
            return errors.Average();
            //grade -= errors.Average();
            //return grade;
        }

        public double Fitness_FailureRate(IList<DataSet> testDataSets)
        {
            var fails = 0;
            var counter = 1;
            foreach (var dataSet in testDataSets)
            {
                ForwardPropagate(false, dataSet.Values);
                if(!OutputLayer.DoesBestCandidateMatch(dataSet.Targets))
                {
                    fails++;
                }
                Console.SetCursorPosition(0, Console.CursorTop);
                Console.Write("Fails/Total: " + fails.ToString("#,###") + "/" + counter.ToString("#,###") + "\t");
                counter++;
            }
            Console.SetCursorPosition(0, Console.CursorTop);
            return (double)fails / (double)testDataSets.Count();
        }
        #endregion

        // Returns a value between -1.0(inclusive) and 1.0 (exclusive)
        public static double GetRandom()
        {
            return 2 * Random.NextDouble() - 1;
        }

        public string Serialize(bool indent = false)
        {
            return JsonConvert.SerializeObject(
                this,
                indent ? Formatting.Indented : Formatting.None,
                new JsonSerializerSettings { PreserveReferencesHandling = PreserveReferencesHandling.Objects }
            );
        }

        public static Network FromJson(string json)
        {
            return JsonConvert.DeserializeObject<Network>(json);
        }
    }
}