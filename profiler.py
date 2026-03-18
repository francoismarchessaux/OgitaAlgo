import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

Z = model.add_all(X)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats("cumtime")
stats.print_stats(20)
