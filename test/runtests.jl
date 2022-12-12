using Styx: Styx, FlowSource, SimpleEMA, SimpleMA, Collector, materialize, flow!, getval, ConditionalCollector, NoFlowSource, setval!, MaxDrawDown
using Test
using BenchmarkTools
using DataStructures

condition(x::Float64) = x > 500_000
condition(::String) = error("Not Defined")

s = FlowSource(Int64)
ema = SimpleEMA(s, 0.10)
ma = SimpleMA(s, 10)
t = (ema, ma);

materialize(t)


simple_test(s, N) = begin
    for i in 1:N
        flow!(s, i)
    end
end

simple_control(N) = begin
    v = CircularBuffer{Int}(10)
    for i in 1:N
        push!(v, i)
    end
end

@btime simple_test(s, 1_000_000)
@btime simple_control(1_000_000)


s = FlowSource(Int64)
mdd = MaxDrawDown(s)



