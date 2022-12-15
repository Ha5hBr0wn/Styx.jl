######################## Using Statements ##########################
using Test
using BenchmarkTools
using Styx: 
    Styx, FlowSource, CumulativeSum, valtype, 
    statetype, materialize, is_val_init, 
    is_state_init, flow!, getval, CumulativeProduct, 
    default, Split, Sum, NoFlowSource, setval!, 
    Collector, Combine



###################### Helper Structures/Functions #########################
struct RealBox{T <: Real} 
    a::T
end

Styx.default(::Type{RealBox{T}}) where T = RealBox{T} |> zero

Base.zero(::Type{RealBox{T}}) where T = RealBox{T}(T |> zero)
Base.one(::Type{RealBox{T}}) where T = RealBox{T}(T |> one)
Base.:+(x::RealBox{T}, y::RealBox{T}) where T = RealBox{T}(x.a + y.a)
Base.:*(x::RealBox{T}, y::RealBox{T}) where T = RealBox{T}(x.a * y.a)
Base.:+(x::RealBox{Int32}, y::RealBox{Float32}) = RealBox{Float16}(x.a + y.a)
Base.:*(x::RealBox{Int32}, y::RealBox{Float32}) = RealBox{Float16}(x.a * y.a)





##################### Node Test Sets ##########################
@testset "CumulativeSum" begin
    s = FlowSource(Int64)
    cs = CumulativeSum(s)

    @test valtype(s) == Int64
    @test statetype(s) == Missing
    @test valtype(cs) == Int64
    @test statetype(cs) == Missing

    materialize([cs])

    @test is_val_init(s) == false
    @test is_val_init(cs) == false
    @test is_state_init(s) == false
    @test is_state_init(cs) == false

    flow!(s, 10)

    @test is_val_init(s) == true
    @test is_val_init(cs) == true
    @test is_state_init(s) == false
    @test is_state_init(cs) == false

    @test getval(s) === 10
    @test getval(cs) === 10

    flow!(s, 5)

    @test getval(s) === 5
    @test getval(cs) === 15

    s2 = FlowSource(Float64)
    cs2 = CumulativeSum(s2)

    @test valtype(s2) == Float64
    @test statetype(s2) == Missing
    @test valtype(cs2) == Float64
    @test statetype(cs2) == Missing

    materialize([cs2])

    @test_throws Exception flow!(s2, 10)

    flow!(s2, 10.0)

    @test getval(cs2) === 10.0
    @test getval(s2) === 10.0

    s3 = FlowSource(String)
    cs3 = CumulativeSum(s3)

    materialize([cs3])

    @test_throws Exception flow!(s3, "Hello")

    s4 = FlowSource(RealBox{Int64})
    cs4 = CumulativeSum(s4)

    materialize([cs4])

    flow!(s4, RealBox(10))

    @test getval(cs4) === RealBox(10)

    flow!(s4, RealBox(5))

    @test getval(cs4) === RealBox(15)
end


@testset "CumulativeProduct" begin
    s = FlowSource(Int64)
    cp = CumulativeProduct(s)

    materialize(cp)

    flow!(s, 10)

    @test getval(cp) === 10

    flow!(s, 2)

    @test getval(cp) === 20

    @test_throws Exception flow!(s, 2.0)

    s2 = FlowSource(RealBox{Float64})
    cp2 = CumulativeProduct(s2)

    materialize(cp2)

    flow!(s2, RealBox(10.0))

    @test getval(cp2) === RealBox(10.0)

    flow!(s2, RealBox(2.0))

    @test getval(cp2) === RealBox(20.0)
end



# @testset "Sum" begin
#     s = FlowSource(Tuple{Int64, Float64})
#     x, y = Split(s)
#     su = Sum(x, y)

#     materialize(su)

#     flow!(s, (2, 3.0))

#     @test getval(x) === 2
#     @test getval(y) === 3.0
#     @test getval(su) === 5.0

#     s2 = FlowSource(Tuple{Int32, Float32})
#     x2, y2 = Split(s2)
#     su2 = Sum(x2, y2)

#     materialize(su2)

#     @test_throws Exception flow!(s2, (2, 3.0))
#     flow!(s2, (2 |> Int32, 3.0 |> Float32))

#     @test getval(su2) === Float32(5.0)

#     s3 = FlowSource(Tuple{RealBox{Int32}, RealBox{Float32}})
#     x3, y3 = Split(s3)
#     su3 = Sum(RealBox{Float32}, x3, y3)

#     materialize(su3)

#     @test_throws Exception flow!(s3, (RealBox(2 |> Int32), RealBox(3.0 |> Float32)))

#     s4 = FlowSource(Tuple{RealBox{Int32}, RealBox{Float32}})
#     x4, y4 = Split(s4)
#     su4 = Sum(RealBox{Float16}, x4, y4)

#     materialize(su4)

#     flow!(s4, (RealBox(2 |> Int32), RealBox(3.0 |> Float32)))

#     @test getval(su4) === RealBox{Float16}(5.0)

#     s5 = FlowSource(Int64)
#     su5 = Sum(s5)

#     materialize(su5)

#     flow!(s5, 10)

#     @test getval(su5) === 10

#     flow!(s5, 5)

#     @test getval(su5) === 5

#     s6 = FlowSource(Tuple{Int64, Float64, Float32})
#     x6, y6, z6 = Split(s6)
#     su6 = Sum(x6, y6, z6)

#     materialize(su6)

#     flow!(s6, (2, 3.0, 10.0 |> Float32))

#     @test getval(su6) === 15.0
#     @test getval(z6) === Float32(10.0)
# end



##################### Node Benchmark Sets ##########################




# Control
control_sum_benchmark(items) = begin
    v = Vector{Float64}()
    for item in items
        push!(v, item[1] + item[2])
    end
    v
end

# Experiment
fs = FlowSource(Tuple{Int64, Float64})
x, y = Split(fs)
s = Sum(x, y)
c = Collector(s)
materialize(c)

flow_all_items!(s, items) = begin
    for item in items
        flow!(s, item)
    end
end

# Setup
N = 1_000_000
items1 = rand(Int64, N)
items2 = rand(Float64, N)
items = [i for i in zip(items1, items2)]

# Benchmark
@time flow_all_items!(fs, items) # 7ms
@time control_sum_benchmark(items); # 5ms