using Styx: 
    FlowSource, SimpleEMA, SimpleMA, 
    Collector, materialize, flow!, 
    getval, ConditionalCollector, NoFlowSource, 
    setval!, MaxDrawDown, Async
using Test
using BenchmarkTools
using DataStructures

@testset 