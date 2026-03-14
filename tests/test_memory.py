from ava.memory import TitansMemory


def test_memory_write_threshold() -> None:
    memory = TitansMemory(max_items=4, write_surprise_threshold=0.5)
    assert not memory.write("too boring", surprise=0.2)
    assert memory.write("important fact", surprise=0.8)


def test_memory_retrieval() -> None:
    memory = TitansMemory(max_items=4, write_surprise_threshold=0.1)
    memory.write("Factorials grow quickly.", surprise=0.7)
    memory.write("The calculator solves arithmetic.", surprise=0.6)
    hits = memory.retrieve("arithmetic", top_k=1)
    assert hits
    assert "calculator" in hits[0].text.lower()
