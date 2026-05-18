from typing import Protocol, runtime_checkable

@runtime_checkable
class Entity:
    entity_id: str

    def update(self, t:float, dt:float) -> None: ...


#this file is used to create a protocol for entities in the simulation. 
# This allows us to define a common interface for all entities, such as targets and radars, 
# without having to define a base class. 