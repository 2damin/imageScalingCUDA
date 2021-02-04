#pragma once

#include <memory>
#include <vector>

class SimpleBBox {
public:
    int x = 0; // (x, y) - top-left corner
    int y = 0;
    int w = 0; // (w, h) - width & height of bounded box
    int h = 0;
    float prob; // confidence - probability that the object was found correctly
    int obj_id; // class of object - from range [0, classes-1]
    bool did_it_move = false;
    std::vector<std::string> color;
    std::vector<std::string> has_hat_bag;
    bool operator==( SimpleBBox A ) {
        return x == A.x && y == A.y && w == A.w && h == A.h && obj_id == A.obj_id;
    }
};

using SimpleBBox = class SimpleBBox;
using SimpleBBoxes = std::vector<SimpleBBox>;
using SimpleBBoxesPtr = std::shared_ptr<SimpleBBoxes>;

class SimpleCProb {
public:
    float prob; // confidence - probability that the object was found correctly
    int obj_id; // class of object - from range [0, classes-1]
};

using SimpleCProb = class SimpleCProb;
using SimpleCProbs = std::vector<SimpleCProb>;
using SimpleCProbsPtr = std::shared_ptr<SimpleCProbs>;
