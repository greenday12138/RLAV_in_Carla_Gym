import collections
import logging
import weakref, math
import carla, time
from enum import Enum


class SemanticTags(Enum):
    NONE=0
    Roads=1
    Sidewalks=2
    Buildings = 3
    Walls = 4
    Fences = 5
    Poles = 6
    TrafficLight = 7
    TrafficSigns = 8
    Vegetation = 9
    Terrain = 10
    Sky = 11
    Pedestrians = 12
    Rider = 13
    Car = 14
    Truck = 15
    Bus = 16
    Train = 17
    Motorcycle = 18
    Bicycle = 19
    Static = 20
    Dynamic = 21
    Other = 22
    Water = 23
    RoadLines = 24
    Ground = 25
    Bridge = 26
    RailTrack = 27
    GuardRail = 28
    Any=255


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class CollisionSensor(object):
    """Class for collision sensors"""

    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_ref = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_ref, event))

    def get_collision_history(self):
        """Get the histroy of collisions"""
        history = collections.defaultdict(int)
        tags = set()
        for tag, frame, intensity in self.history:
            history[frame] += intensity
            tags.add(tag)
        return history, tags

    def clear_history(self):
        self.history.clear()

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        logging.info('Collision with %r',actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        for tag in event.other_actor.semantic_tags:
            self.history.append((SemanticTags(tag), event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor) -> None:
        self.sensor = None
        self._parent = parent_actor
        self.count = 0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    def get_invasion_count(self):
        return self.count

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        self.count += 1
        lane_type = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_type]
        # logging.info('Crossed line %s' % ' and '.join(text))
