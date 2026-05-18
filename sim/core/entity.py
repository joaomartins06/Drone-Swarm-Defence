from typing import Protocol, runtime_checkable

''' 
Defines the Entity protocol: the structural contract that every world object 
(target, radar, interceptor) must satisfy to be stepped forward by the World.
'''

@runtime_checkable
class Entity(Protocol):
    entity_id: str

    def update(self, t:float, dt:float) -> None: ...

    