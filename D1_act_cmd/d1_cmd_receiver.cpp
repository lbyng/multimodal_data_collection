#include <zmq.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include "msg/ArmString_.hpp"

#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

using namespace unitree::robot;
using namespace unitree::common;
using json = nlohmann::json;

#define TOPIC "rt/arm_Command"

int main() {
    // Init Unitree DDS publisher
    ChannelFactory::Instance()->Init(0);
    ChannelPublisher<unitree_arm::msg::dds_::ArmString_> publisher(TOPIC);
    publisher.InitChannel();

    // Init ZeroMQ SUB socket
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::sub);
    socket.connect("tcp://localhost:5555");
    socket.set(zmq::sockopt::subscribe, "");

    std::cout << "Listening to joint angles on tcp://localhost:5555..." << std::endl;

    while (true) {
        zmq::message_t msg;
        socket.recv(msg, zmq::recv_flags::none);

        std::string msg_str(static_cast<char*>(msg.data()), msg.size());
        try {
            json parsed = json::parse(msg_str);
            if (!parsed.contains("positions")) continue;
            auto angles = parsed["positions"];
            if (angles.size() != 7) continue;

            // Compose command JSON for Unitree
            json cmd = {
                {"seq", 4},
                {"address", 1},
                {"funcode", 2},
                {"data", {
                    {"mode", 0},
                    {"angle0", angles[0]},
                    {"angle1", angles[1]},
                    {"angle2", angles[2]},
                    {"angle3", angles[5]},
                    {"angle4", angles[3]},
                    {"angle5", angles[4]},
                    {"angle6", angles[6]}
                }}
            };

            // Wrap into DDS ArmString_ message
            unitree_arm::msg::dds_::ArmString_ arm_msg{};
            arm_msg.data_() = cmd.dump();
            publisher.Write(arm_msg);

            std::cout << "Sent to D1: " << cmd.dump() << std::endl;

        } catch (std::exception& e) {
            std::cerr << "JSON parse/send error: " << e.what() << std::endl;
        }
    }

    return 0;
}
