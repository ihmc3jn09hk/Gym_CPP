#include "gym_torch.h"

CartPole::~CartPole()
{

}

at::Tensor CartPole::reset()
{
   auto tmp = torch::zeros({4,});
   mState = tmp.uniform_(-0.05,0.05);
   //state[0] = state[1] = state[2] = state[3] = 0.02;

   steps_beyond_done = -1;
   return mState;
}

Gym_Torch::dType CartPole::step(at::Tensor action)
{
    double x = mState[0].item().toDouble();
    double x_dot = mState[1].item().toDouble();
    double theta = mState[2].item().toDouble();
    double theta_dot = mState[3].item().toDouble();

//    std::cout << action << " - " << action.item().toInt() << std::endl;

    auto force = action.item().toInt() == 1 ? force_mag : -force_mag;
    auto costheta = std::cos(theta);
    auto sintheta = std::sin(theta);

    /** For the interested reader:
    * https://coneural.org/florian/papers/05_cart_pole.pdf
    * **/
    auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta ) / total_mass;
    auto thetaAcc = (gravity * sintheta - costheta * temp) /
            (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
    auto xacc = temp - polemass_length * thetaAcc * costheta / total_mass;

    if ( kinematics_integrator == "euler") {
        x = x + tau * x_dot;
        x_dot = x_dot + tau * xacc;
        theta = theta + tau * theta_dot;
        theta_dot = theta_dot + tau * thetaAcc;
    } else {  // semi-implicit euler
        x_dot = x_dot + tau * xacc;
        x = x + tau * x_dot;
        theta_dot = theta_dot + tau * thetaAcc;
        theta = theta + tau * theta_dot;
    }

    mState[0] = x;
    mState[1] = x_dot;
    mState[2] = theta;
    mState[3] = theta_dot;

    auto _done = (x < -x_threshold) || (x > x_threshold) ||
            (theta < -theta_threshold_radians) || (theta > theta_threshold_radians);

    auto reward = torch::zeros({1});
    auto tmp = torch::Tensor();
    auto done = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt));
    done[0] = _done ? 1 : 0;

    if (!_done) {
        reward[0] = 1.0;
    } else if ( 0 > steps_beyond_done ) {//Pole just fell!
        steps_beyond_done = 0;
        reward[0] = 1.0;
    } else {
        if (steps_beyond_done == 0){
            std::cout <<
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            << std::endl;
            steps_beyond_done += 1;
            reward[0] = 0.0;
        }
    }
    return std::make_tuple<>(mState, reward, done, tmp);
 }

at::Tensor CartPole::sample_action()
{
    return torch::randint(0,2,{1});
}

int CartPole::action_dimension()
{
    return 1;
}

int CartPole::state_dimension()
{
    return 4;
}

CartPole_Continous::CartPole_Continous(bool b2D)
    :m_b2D(b2D)
{

}

CartPole_Continous::~CartPole_Continous()
{
    std::cout << "CartPole_Continous" << std::endl;
}

at::Tensor CartPole_Continous::reset()
{
    auto tmp = torch::zeros({ state_dimension(),});
    mState = tmp.uniform_(-0.05,0.05);
    //state[0] = state[1] = state[2] = state[3] = 0.02;

    steps_beyond_done = -1;
    return mState;
}

Gym_Torch::dType CartPole_Continous::step(at::Tensor action)
{
    const int stateDim = state_dimension();
    auto reward = torch::zeros({1});
    auto tmp = torch::Tensor();
    auto done = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt));
    bool _done = false;

    for( int i=0; i<stateDim; i+=4 ) {
        double x = mState[i].item().toDouble();
        double x_dot = mState[i+1].item().toDouble();
        double theta = mState[i+2].item().toDouble();
        double theta_dot = mState[i+3].item().toDouble();
        double x_tip = 0.0;

    //    std::cout << action << " - " << action.item().toInt() << std::endl;

        auto force = action[i/4].item().toDouble() * force_mag;
        auto costheta = std::cos(theta);
        auto sintheta = std::sin(theta);

        /** For the interested reader:
        * https://coneural.org/florian/papers/05_cart_pole.pdf
        * **/
        auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta ) / total_mass;
        auto thetaAcc = (gravity * sintheta - costheta * temp) /
                (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        auto xacc = temp - polemass_length * thetaAcc * costheta / total_mass;

        if ( kinematics_integrator == "euler") {
            x = x + tau * x_dot;
            x_dot = x_dot + tau * xacc;
            theta = theta + tau * theta_dot;
            theta_dot = theta_dot + tau * thetaAcc;
            x_tip = 2*length * std::sin(theta) + x;
        } else {  // semi-implicit euler
            x_dot = x_dot + tau * xacc;
            x = x + tau * x_dot;
            theta_dot = theta_dot + tau * thetaAcc;
            theta = theta + tau * theta_dot;
            x_tip = 2*length * std::sin(theta) + x;
        }

        mState[i] = x;
        mState[i+1] = x_dot;
        mState[i+2] = theta;
        mState[i+3] = theta_dot;

        _done |= (x < -x_threshold) || (x > x_threshold) ||
            (theta < -theta_threshold_radians) || (theta > theta_threshold_radians)
                || (x_tip < -x_threshold) || (x_tip > x_threshold);
    }

    done[0] = _done ? 1 : 0;
    if (!_done) {
        reward[0] = 1.0;
    } else if ( 0 > steps_beyond_done ) {//Pole just fell!
        steps_beyond_done = 0;
        reward[0] = 1.0;
    } else {
        if (steps_beyond_done == 0){
            std::cout <<
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            << std::endl;
            steps_beyond_done += 1;
            reward[0] = 0.0;
        }
    }

    return std::make_tuple<>(mState, reward, done, tmp);
}

at::Tensor CartPole_Continous::sample_action()
{
    if ( m_b2D ) {
        return torch::normal(0.0, 0.5, {2});
    }
    return torch::normal(0.0, 0.5, {1});
}

int CartPole_Continous::action_dimension()
{
    return m_b2D ? 2 : 1;
}

int CartPole_Continous::state_dimension()
{
    return m_b2D ? 8 : 4;
}

CartPole_ContinousVision::CartPole_ContinousVision(bool b2D, int preFramesCount)
    :CartPole_Continous(b2D)
    ,mPreFramesCount(preFramesCount)
{
    mRenderCB = nullptr;
}

CartPole_ContinousVision::~CartPole_ContinousVision()
{
    mRenderCB = nullptr;
}

at::Tensor CartPole_ContinousVision::reset()
{
    auto tmp = torch::zeros({ CartPole_Continous::state_dimension(),});
    mState = tmp.uniform_(-0.05,0.05);
    //state[0] = state[1] = state[2] = state[3] = 0.02;

    steps_beyond_done = -1;

    mvPreFrames.clear();

    //Create an image
    if ( mRenderCB ) {
        std::vector<double> pos = {mState[0].item().toDouble(), mState[4].item().toDouble()};
        std::vector<double> ang = {mState[2].item().toDouble(), mState[6].item().toDouble()};
        std::vector<unsigned int> rgba;
        auto &&[w, h] = (*mRenderCB)(pos, ang, rgba);
        //WARNING: No error check for "rgba"
        auto imgTensor = torch::from_blob(rgba.data(), {w * h, 4},
                                          torch::TensorOptions().dtype(torch::kInt8));

        imgTensor = imgTensor.index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                                     torch::indexing::Slice(2,4)}).clone().toType(torch::kFloat);

        for ( int i=0; i<mPreFramesCount; ++i ) {
            mvPreFrames.push_back(imgTensor.clone());
        }
        std::vector<torch::Tensor> stack(mPreFramesCount+1);
        std::copy(mvPreFrames.begin(), mvPreFrames.end(), stack.begin());
        stack[mPreFramesCount] = imgTensor;

        torch::Tensor state_Img = torch::stack(stack, 1);
        return state_Img.view({-1});
    }
    return mState;
}

//#include <QImage>
//#include <QThread>
Gym_Torch::dType CartPole_ContinousVision::step(at::Tensor action)
{
    const int stateDim = CartPole_Continous::state_dimension();
    auto reward = torch::zeros({1});
    auto tmp = torch::Tensor();
    auto done = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt));
    bool _done = false;
    auto _extra_reward = 0.0;

    for( int i=0; i<stateDim; i+=4 ) {
        double x = mState[i].item().toDouble();
        double x_dot = mState[i+1].item().toDouble();
        double theta = mState[i+2].item().toDouble();
        double theta_dot = mState[i+3].item().toDouble();
        double x_tip = 0.0;

        double last_theta = theta;

        auto force = action[i/4].item().toDouble() * force_mag;
        auto costheta = std::cos(theta);
        auto sintheta = std::sin(theta);

        /** For the interested reader:
        * https://coneural.org/florian/papers/05_cart_pole.pdf
        * **/
        auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta ) / total_mass;
        auto thetaAcc = (gravity * sintheta - costheta * temp) /
                (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        auto xacc = temp - polemass_length * thetaAcc * costheta / total_mass;

        if ( kinematics_integrator == "euler") {
            x = x + tau * x_dot;
            x_dot = x_dot + tau * xacc;
            theta = theta + tau * theta_dot;
            theta_dot = theta_dot + tau * thetaAcc;
            x_tip = 2*length * std::sin(theta) + x;
        } else {  // semi-implicit euler
            x_dot = x_dot + tau * xacc;
            x = x + tau * x_dot;
            theta_dot = theta_dot + tau * thetaAcc;
            theta = theta + tau * theta_dot;
        }

        if ( x < -x_threshold ) {
            x = 2*x_threshold + x;
        } else if ( x > x_threshold) {
            x = x - 2*x_threshold;
        }

        mState[i] = x;
        mState[i+1] = x_dot;
        mState[i+2] = theta;
        mState[i+3] = theta_dot;

        _done |= (theta < -theta_threshold_radians) || (theta > theta_threshold_radians)
/*                || (x < -x_threshold) || (x > x_threshold)
                || (x_tip < -x_threshold) || (x_tip > x_threshold)*/;

        if ( std::abs(theta) - std::abs(last_theta) < 0.0 ) {
            _extra_reward += 0.2 * (theta_threshold_radians - std::abs(theta));
        }
    }

    done[0] = _done ? 1 : 0;
    if (!_done) {
        reward[0] = 1.0 + _extra_reward;
    } else if ( 0 > steps_beyond_done ) {//Pole just fell!
        steps_beyond_done = 0;
        reward[0] = 1.0 + _extra_reward;
    } else {
        if (steps_beyond_done == 0){
            std::cout <<
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            << std::endl;
            steps_beyond_done += 1;
            reward[0] = 0.0;
        }
    }

    //Create an image
    if ( mRenderCB ) {
        std::vector<double> pos = {mState[0].item().toDouble(), mState[4].item().toDouble()};
        std::vector<double> ang = {mState[2].item().toDouble(), mState[6].item().toDouble()};
        std::vector<unsigned int> rgba;
        auto &&[w, h] = (*mRenderCB)(pos, ang, rgba);
        //WARNING: No error check for "rgba"
        auto imgTensor = torch::from_blob(rgba.data(), {w * h, 4},
                                          torch::TensorOptions().dtype(torch::kInt8));

        imgTensor = imgTensor.clone().toType(torch::kFloat);

//      if ( 0.523 < std::fabs(mState[2].item().toDouble())){
//            std::cout << "GG" << std::endl;
//            auto tmpTest = imgTensor.clone().toType(torch::kInt8);
//            tmpTest = tmpTest.reshape({w, h, 4});
//            QImage tmpImg((const uchar*)tmpTest.data_ptr(), w, h, QImage::Format_ARGB32);
//            if ( tmpImg.isNull()) {
//                std::exit(-1);
//            }
//            tmpImg.save("fromTensor.png");
//            tmpTest = lastImg.clone().toType(torch::kInt8);
//            tmpTest = tmpTest.reshape({w, h, 4});
//            tmpImg = QImage((const uchar*)tmpTest.data_ptr(), w, h, QImage::Format_ARGB32);
//            if ( tmpImg.isNull()) {
//                std::exit(-1);
//            }
//            tmpImg.save("fromTensor_last.png");
            //QThread::currentThread()->sleep(100000);
//        }

        imgTensor = imgTensor.index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                                     torch::indexing::Slice(2,4)});

        if ( int(mvPreFrames.size()) < mPreFramesCount ) {  //The env has been reset()
            std::cout << "You are calling 'step()' before reset() the environment."
                         "No previous frame/history recorded!"
                      << std::endl;
        }

        std::vector<torch::Tensor> stack(mPreFramesCount+1);
        std::copy(mvPreFrames.begin(), mvPreFrames.end(), stack.begin());
        stack[mPreFramesCount] = imgTensor;

        torch::Tensor state_Img = torch::stack(stack, 1);

//        if ( 0.523 < std::fabs(mState[2].item().toDouble())) {
//            std::cout << state_Img.sizes() << std::endl;
//            auto test = state_Img.index({0,
//                        torch::indexing::Slice(torch::indexing::None,torch::indexing::None),
//                                         0
//                        }).clone();

//            test = torch::stack({test,
//                                 torch::zeros({w * h}),
//                                 torch::zeros({w * h}),
//                                 test}, 1).toType(torch::kInt8);
//            std::cout << test.sizes() << std::endl;
//            auto testimg1 = QImage((const uchar*)test.data_ptr(), w, h, QImage::Format_ARGB32);
//            if ( testimg1.isNull()) {
//                std::exit(-1);
//            }
//            testimg1.save("fromTensor_BD_last.png");
//            test = state_Img.index({1,
//                    torch::indexing::Slice(torch::indexing::None,torch::indexing::None),
//                                    0
//                    }).clone();
//            test = torch::stack({torch::zeros({w * h}),
//                                 test,
//                                 torch::zeros({w * h}),
//                                 test}, 1).toType(torch::kInt8);
//            auto testimg2 = QImage((const uchar*)test.data_ptr(), w, h, QImage::Format_ARGB32);
//            if ( testimg2.isNull()) {
//                std::exit(-1);
//            }
//            testimg2.save("fromTensorBD.png");
//            QThread::currentThread()->sleep(100000);
//        }

        mvPreFrames.push_back(imgTensor.clone());
        if ( int(mvPreFrames.size()) > mPreFramesCount) {
            mvPreFrames.pop_front();
        }

        return std::make_tuple<>(state_Img.view({-1}), reward, done, tmp);
    } else {
        std::cout << "No renderer provided. Data-level \"state\" will be returned." << std::endl;
    }
    return std::make_tuple<>(mState, reward, done, tmp);
}

void CartPole_ContinousVision::setRender_Callback(std::function<std::pair<int,int> (std::vector<double>,
                                                                      std::vector<double>,
                                                                      std::vector<unsigned int>&)> *cb)
{
    mRenderCB = cb;
}

int CartPole_ContinousVision::state_dimension()
{
    //W * H * int(RGB-D)
    return 128*128*2*2;
}
