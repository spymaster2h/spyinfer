#include <iostream>
#include <string>
#include <vector>

#include "engine/llm_engine.hpp"
#include "utils/tokenizer.hpp"
#include "utils/precision.hpp"


int main(int argc, char* argv[])
{
    std::string model_path;
    std::string backend_type = "cpu"; 
    
    int opt;
    while ((opt = getopt(argc, argv, "m:b:h")) != -1) {
        switch (opt) {
            case 'm':
                model_path = optarg;
                break;
            case 'b':
                backend_type = optarg;
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " -m <model_path> [-b <backend_type>]" << std::endl;
                std::cout << "Available backends: cpu, cuda" << std::endl;
                return 0;
            default:
                std::cerr << "Usage: " << argv[0] << " -m <model_path> [-b <backend_type>]" << std::endl;
                std::cerr << "Use -h for help" << std::endl;
                return 1;
        }
    }

    if (model_path.empty())
    {
        std::cerr << "Usage: " << argv[0] << " -m <model_path> [-b <backend_type>]" << std::endl;
        std::cerr << "Use -h for help" << std::endl;
        return 1;
    }


    // 检查后端类型是否有效
    if (backend_type != "cpu" && backend_type != "cuda") {
        std::cerr << "Invalid backend type: " << backend_type << std::endl;
        std::cerr << "Available backends: cpu, cuda" << std::endl;
        return 1;
    }

#ifdef USE_CUDA
    if (backend_type == "cuda") {
        std::cout << "Using CUDA backend" << std::endl;
    } else {
        std::cout << "Using CPU backend" << std::endl;
    }
#else
    if (backend_type == "cuda") {
        std::cerr << "CUDA backend not compiled in. Rebuild with -DUSE_CUDA flag." << std::endl;
        return 1;
    }
    std::cout << "Using CPU backend" << std::endl;
#endif



    spyinfer::LLMEngine engine(model_path, {{"backend_type", backend_type}});

    int conversation_id = -1;

    while (true)
    {
        std::cout << "\nUser: ";
        std::string user_input;
        std::getline(std::cin, user_input);

        if (user_input == "/exit")
        {
            if (conversation_id != -1) {
                engine.remove_request(conversation_id);
            }
            break;
        }

        if (user_input == "/new")
        {
            if (conversation_id != -1) {
               engine.remove_request(conversation_id);
            }
            conversation_id = -1;
            std::cout << "Starting new conversation." << std::endl;
            continue;
        }


        if (conversation_id == -1)
        {
            conversation_id = engine.add_request({{"user", user_input}});
            std::cout << "Assistant: " << std::flush;
        }
        else
        {
            // Subsequent turn in an existing conversation
            engine.append_to_request(conversation_id, {"user", user_input});
            std::cout << "Assistant: " << std::flush;
        }

        // Streaming loop
        std::string printed_output = "";
        while (!engine.is_request_finished(conversation_id))
        {
            engine.step();
            // Get only the generated part, not the full history
            std::string current_response = engine.get_output(conversation_id);
            if (current_response.length() > printed_output.length())
            {
                std::string new_text = current_response.substr(printed_output.length());
                std::cout << new_text << std::flush;
                printed_output = current_response;
            }
        }
        std::cout << std::endl;
    }

    return 0;
}