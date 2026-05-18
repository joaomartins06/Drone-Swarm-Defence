from dataclasses import dataclass, field
from typing import Callable
from sim.core import Entity
from sim.core.clock import Clock
from sim.core.entity import Entity


TickHook = Callable[['World'], None]

@dataclass
class World:
    #default_factory allows to ahve a new clock for each world instance
    clock: Clock = field(default_factory=Clock)
    entities: list[Entity] = field(default_factory=list)
    #list of callables to be called after on measurement ticks
    on_measurement_hooks: list[TickHook] = field(default_factory=list)
    #same thing on render ticks
    on_render_hooks: list[TickHook] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        #check if the entity id is unique in the world
        existing_ids = {e.entity_id for e in self.entities}
        if entity.entity_id in existing_ids:
            raise ValueError(f"Entity with id {entity.entity_id} already exists in the world")
        
        self.entities.append(entity)

    def get_entity(self, entity_id: str) -> Entity:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
            
        raise KeyError(f"Entity with id {entity_id} not found in the world")
    
    def step(self) -> None:
        #advance the clock
        self.clock.advance()

        t = self.clock.t
        dt = self.clock.dt

        #update all entities in the world
        for entity in self.entities:
            entity.update(t, dt)

        #call measurement hooks if it's a measurement tick
        if self.clock.is_measurement_tick():
            for hook in self.on_measurement_hooks:
                #recall that on_measurement_hooks is a list of callables
                hook(self)

        #call render hooks if it's a render tick
        if self.clock.is_render_tick():
            for hook in self.on_render_hooks:
                hook(self)

    def run_for(self, duration_s: float) -> None:

        target_tick_count = self.clock.tick_count + round(duration_s / self.clock.dt)
        while self.clock.tick_count < target_tick_count:
            self.step()
    