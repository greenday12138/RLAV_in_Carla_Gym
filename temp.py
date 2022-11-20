import numpy as np
import random,logging
import carla
import torch

class Temp:
    def __init__(self,args) -> None:
        logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)
        vehicles_id_list=[]

        self.client=carla.Client(args.host,args.port)
        self.client.set_timeout(5.0)
        self.client.load_world('Town01')
        self.synchronous_master=False
        self.sync=args.sync
        self.tm_port=args.tm_port
        self.vehicles_id_list=[]
        self.num_of_vehicles=args.num_of_vehicles
        camera=None

        self.world=self.client.get_world()
        origin_settings=self.world.get_settings()
        self._spawn()

        
    def _spawn(self):
        traffic_manager=self.client.get_trafficmanager(self.tm_port)
        # every vehicle keeps a distance of 3.0 meter
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        # Set physical mode only for cars around ego vehicle to save computation
        traffic_manager.set_hybrid_physics_mode(True)
        # default speed limit is 54
        traffic_manager.global_percentage_speed_difference(-80)

        if self.sync:
            print('temp')
            settings=self.world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                self.synchronous_master=True
                settings.synchronous_mode=True
                settings.fixed_delta_seconds=0.05   #20 fps
                self.world.apply_settings(settings)
        
        blueprints_vehicle=self.world.get_blueprint_library().filter('vehicle.*')
        # sort the vehicle list by id
        blueprints_vehicle=sorted(blueprints_vehicle,key=lambda bp:bp.id)

        spawn_points=self.world.get_map().get_spawn_points()
        num_of_spawn_points=len(spawn_points)

        if self.num_of_vehicles<num_of_spawn_points:
            random.shuffle(spawn_points)
        elif self.num_of_vehicles>=num_of_spawn_points:
            msg='requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg,self.num_of_vehicles,num_of_spawn_points)
            self.num_of_vehicles=num_of_spawn_points-1
        
        # Use command to apply actions on batch of data
        SpawnActor=carla.command.SpawnActor
        SetAutopilot=carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor #FutureActor is eaqual to 0

        command_batch=[]

        for n,transform in enumerate(spawn_points):
            if n>=self.num_of_vehicles:
                break
                
            blueprint=random.choice(blueprints_vehicle)
            if blueprint.has_attribute('color'):
                color=random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color',color)
            if blueprint.has_attribute('driver_id'):
                driver_id=random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id',driver_id)

            #set a attribute indicating autopilot mode
            blueprint.set_attribute('role_name','autopilot')
            #spawn the cars and their autopilot all rogether
            command_batch.append(SpawnActor(blueprint,transform).
                then(SetAutopilot(FutureActor,True,traffic_manager.get_port())))  #the 0 here doesn't mean anything

        #excute the command batch
        for (i,response) in enumerate(self.client.apply_batch_sync(command_batch,self.synchronous_master)):
            if response.has_error():
                logging.error(response.error)
            else:
                print('Future Actor',response.actor_id)
                self.vehicles_id_list.append(response.actor_id)
                #traffic_manager.ignore_lights_percentage(world.get_actor(response.actor_id),100)
        
        vehicles_list=self.world.get_actors().filter('vehicle.*')
        # wait for a tick to ensure client receives the last transform of the vehicles we have just created
        # if not args.sync or not self.synchronous_master:
        #     """处于异步模式的客户首先通过world.wait_for_tick() 等待server 更新，
        #     一旦更新了它们会立刻通过world.on_tick 里的callback 来提取这个更新的wordsnapshot里面的信息,如timestamp"""
        #     self.world.wait_for_tick()
        # else:
        #     """tick函数让server的simulation更新一次"""
        #     self.world.tick()

        #set several of the cars as dangerous car
        for i in range(2):
            danger_car=vehicles_list[i]
            #crazy car ignore traffic light, do not keep safe distance and very fast
            traffic_manager.ignore_lights_percentage(danger_car,100)
            traffic_manager.distance_to_leading_vehicle(danger_car,0)
            traffic_manager.vehicle_percentage_speed_difference(danger_car,-50)
        
        print('spawned %d vehicles, press Ctrl+C to exit.'%(len(vehicles_list)))

        # #create ego vehicle
        # ego_vehicle_bp=world.get_blueprint_library().find('vehicle.audi.a2')
        # #green color
        # ego_vehicle_bp.set_attribute('color','0,255,0')
        # #set this one as ego
        # ego_vehicle_bp.set_attribute('role_name','hero')
        # #get a valid transform that has not been assigned yet
        # transform=spawn_points[len(vehicles_id_list)]

        # ego_vehicle=world.spawn_actor(ego_vehicle_bp,transform)
        # ego_vehicle.set_autopilot(True,args.tm_port)

        # sensor_queue=Queue(maxsize=10)
        # camera_bp=world.get_blueprint_library().find('sensor.camera.rgb')
        # camera_transform=carla.Transform(carla.Location(x=1.5,z=2.4))
        # camera=world.spawn_actor(camera_bp,camera_transform,attach_to=ego_vehicle)
        # camera.listen(lambda image:sensor_callback(image,sensor_queue))

        #simulation without sensor version

    def step(self):
        if self.synchronous_master:
            self.world.tick()
        else:
            self.world.wait_for_tick()

if __name__=='__main__':
    # arr1=torch.tensor([[1],[-2]],dtype=torch.float32)
    # arr2=arr1.clone().detach()
    # for i in range(arr1.shape[0]):
    #     if arr1[i][0]<0:
    #         arr1[i][0]=0
    #     if arr2[i][0]>=0:
    #         arr2[i][0]=0
    # arr3=torch.cat((arr1,arr2),dim=1)
    # arr=np.array([[1,-1],[-1,-1]])
    # arr[:,0]+=arr[:,0]
    # sq=torch.tensor([[1]],dtype=torch.float32)
    # tt=torch.tensor(True,dtype=torch.float32)
    # tf=torch.tensor(False,dtype=torch.float32)
    #
    # # print(tt,tf,sep='\t')
    # # print(torch.squeeze(sq))
    # print(arr1,arr2,arr3,sep='\n')
    # print(torch.split(arr3,split_size_or_sections=[1,1],dim=1),sep='\n')
    a=0.5
    for i in range(20000):
        a=a*0.9999
    print(a)