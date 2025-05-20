#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/common/time/time_tool.hpp>
#include "msg/PubServoInfo_.hpp"
#include "msg/ArmString_.hpp"
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <iostream>

#define TOPIC "current_servo_angle"
#define TOPIC1 "rt/arm_Feedback"

using namespace unitree::robot;
using namespace unitree::common;
using json = nlohmann::json;

// ZeroMQ context and socket as global variables
zmq::context_t context(1);
zmq::socket_t publisher(context, zmq::socket_type::pub);

void Handler(const void* msg)
{
    const unitree_arm::msg::dds_::ArmString_* pm = (const unitree_arm::msg::dds_::ArmString_*)msg;
    std::string feedback_data = pm->data_();
    
    // std::cout << "armFeedback_data:" << feedback_data << std::endl;
    
    try {
        json feedback_json = json::parse(feedback_data);
        
        if (feedback_json.contains("seq") && 
            feedback_json.contains("address") && 
            feedback_json["address"] == 2 && 
            feedback_json.contains("funcode") && 
            feedback_json["funcode"] == 1 && 
            feedback_json.contains("data")) {
            
            auto& data = feedback_json["data"];
            
            if (data.contains("angle0")) {
                std::vector<double> angles;
                for (int i = 0; i <= 6; i++) {
                    std::string angle_key = "angle" + std::to_string(i);
                    if (data.contains(angle_key)) {
                        angles.push_back(data[angle_key].get<double>());
                    }
                }
                
                json position_msg = {
                    {"joint_positions", angles}
                };
                
                std::string msg_str = position_msg.dump();
                zmq::message_t zmq_msg(msg_str.size());
                memcpy(zmq_msg.data(), msg_str.c_str(), msg_str.size());
                publisher.send(zmq_msg, zmq::send_flags::none);
                
                // std::cout << msg_str << std::endl;
            }
        }
    } catch (std::exception& e) {
        std::cerr << "处理反馈数据时出错: " << e.what() << std::endl;
    }
}

int main()
{   
    publisher.bind("tcp://*:5556");
    std::cout << "ZeroMQ发布者绑定到tcp://*:5556" << std::endl;
    
    ChannelFactory::Instance()->Init(0);
    ChannelSubscriber<unitree_arm::msg::dds_::ArmString_> subscriber(TOPIC1);
    subscriber.InitChannel(Handler);
    
    std::cout << "监听Unitree机械臂反馈并通过ZeroMQ发布关节位置..." << std::endl;

    while (true)
    {
        sleep(10);
    }

    return 0;
}