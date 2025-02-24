/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011-2018 Centre Tecnologic 
 * de Telecomunicacions de Catalunya (CTTC)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: 
 *    - Jaume Nin <jaume.nin@cttc.cat>
 *    - Manuel Requena <manuel.requena@cttc.es>
 * Modified by:
 *    - Pengyu Liu <eic_lpy@hust.edu.cn>
 *    - Hao Yin <haoyin@uw.edu>
 *    - Muyuan Shen <muyuan_shen@hust.edu.cn>
 *
 * Further modifications for demonstration of multi-flow, 
 * multi-enb scenario (5G-V2X-like).
 */

 #include "ns3/applications-module.h"
 #include "ns3/config-store-module.h"
 #include "ns3/core-module.h"
 #include "ns3/flow-monitor-module.h"
 #include "ns3/internet-module.h"
 #include "ns3/lte-module.h"
 #include "ns3/mobility-module.h"
 #include "ns3/point-to-point-module.h"
 
 using namespace ns3;
 using namespace std;
 
 NS_LOG_COMPONENT_DEFINE("LenaV2XScenario");
 
 int
 main(int argc, char* argv[])
 {
     // --- Simulation parameters ---
     uint16_t nUePerEnb = 5;          // each eNB has how many UEs (vehicles)
     uint32_t nEnb = 2;               // number of eNB
     double simTime = 5.0;            // total simulation time (s)
     string outputCsv = "lte_cqi_result.csv"; // CSV output file
     bool enableFlowMonitor = true;   // set false to skip FlowMonitor
 
     // "Vehicle" speed
     double vehicleSpeed = 33.33; // ~120 km/h
     // Alternatively, 50.0 => 180 km/h
 
     // Command line
     CommandLine cmd;
     cmd.AddValue("nUePerEnb", "Number of UEs (vehicles) per eNB", nUePerEnb);
     cmd.AddValue("nEnb", "Number of eNB (base stations)", nEnb);
     cmd.AddValue("simTime", "Simulation time (s)", simTime);
     cmd.AddValue("outputCsv", "CSV filename for flow results", outputCsv);
     cmd.AddValue("vehicleSpeed", "Vehicle speed (m/s)", vehicleSpeed);
     cmd.AddValue("enableFlowMonitor", "Enable FlowMonitor", enableFlowMonitor);
     cmd.Parse(argc, argv);
 
     ConfigStore inputConfig;
     inputConfig.ConfigureDefaults();
     cmd.Parse(argc, argv);
 
     // Basic random seed
     RngSeedManager::SetSeed(6);
     RngSeedManager::SetRun(4);
 
     // Create LTE and EPC
     Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
     Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
     lteHelper->SetEpcHelper(epcHelper);
     // Use custom Round-Robin scheduler that calls into ns3-ai
     lteHelper->SetSchedulerType("ns3::MyRrMacScheduler");
     lteHelper->SetAttribute("PathlossModel", StringValue("ns3::FriisSpectrumPropagationLossModel"));
 
     // PGW
     Ptr<Node> pgw = epcHelper->GetPgwNode();
 
     // Create remote host
     NodeContainer remoteHostContainer;
     remoteHostContainer.Create(1);
     Ptr<Node> remoteHost = remoteHostContainer.Get(0);
     InternetStackHelper internet;
     internet.Install(remoteHostContainer);
 
     // Use a big fat pipe to connect PGW and RemoteHost
     PointToPointHelper p2ph;
     p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
     p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
     p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
     NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
     Ipv4AddressHelper ipv4h;
     ipv4h.SetBase("1.0.0.0", "255.0.0.0");
     Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
     // interface 0 is localhost, 1 is the p2p device
 
     Ipv4StaticRoutingHelper ipv4RoutingHelper;
     Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
         ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
     remoteHostStaticRouting->AddNetworkRouteTo(
         Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);
 
     // Create eNB nodes
     NodeContainer enbNodes;
     enbNodes.Create(nEnb);
 
     // Create UE nodes
     NodeContainer ueNodes;
     ueNodes.Create(nEnb * nUePerEnb);
 
     // Position eNBs
     Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();
     // e.g. place each eNB 500m apart on x-axis
     double distEnb = 500.0; 
     for (uint32_t i = 0; i < nEnb; ++i)
     {
         enbPositionAlloc->Add(Vector(i * distEnb, 0.0, 20.0)); // height = 20m
     }
     MobilityHelper enbMobility;
     enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
     enbMobility.SetPositionAllocator(enbPositionAlloc);
     enbMobility.Install(enbNodes);
 
     // Position UEs (vehicles)
     //   Suppose each eNB has nUePerEnb vehicles
     //   We'll place them initially behind the eNB, 
     //   then let them move in +x direction at 'vehicleSpeed'
     Ptr<ListPositionAllocator> uePositionAlloc = CreateObject<ListPositionAllocator>();
     // let's space them out 50m behind the eNB
     // e.g. if eNB i is at x_i, then the j-th UE is around x_i - 300, 
     // with some small offset in y
     for (uint32_t i = 0; i < nEnb; ++i)
     {
         double enbX = i * distEnb;
         for (uint32_t j = 0; j < nUePerEnb; ++j)
         {
             double offsetY = (j % 2 == 0) ? 20.0 : -20.0; // just some variation
             uePositionAlloc->Add(Vector(enbX - 300.0, offsetY, 1.5)); // vehicle height=1.5
         }
     }
 
     MobilityHelper ueMobility;
     ueMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
     ueMobility.SetPositionAllocator(uePositionAlloc);
     ueMobility.Install(ueNodes);
 
     // Set velocity
     for (uint32_t idx = 0; idx < ueNodes.GetN(); idx++)
     {
         Ptr<ConstantVelocityMobilityModel> mobilityModel =
             ueNodes.Get(idx)->GetObject<ConstantVelocityMobilityModel>();
         // move along x direction
         Vector speedVec(vehicleSpeed, 0.0, 0.0);
         mobilityModel->SetVelocity(speedVec);
     }
 
     // Install LTE Devices
     NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
     NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);
 
     // Some basic PHY config
     for (uint32_t i = 0; i < nEnb; ++i)
     {
         Ptr<LteEnbNetDevice> enbLteDev = enbLteDevs.Get(i)->GetObject<LteEnbNetDevice>();
         Ptr<LteEnbPhy> enbPhy = enbLteDev->GetPhy();
         enbPhy->SetAttribute("TxPower", DoubleValue(30.0));
         enbPhy->SetAttribute("NoiseFigure", DoubleValue(5.0));
     }
 
     // Install IP stack on UEs
     internet.Install(ueNodes);
     Ipv4InterfaceContainer ueIpIfaces = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));
 
     // Attach UE to eNB (Round-robin or nearest eNB in real scenario; here let's do all in i-th group to i-th eNB)
     for (uint32_t i = 0; i < nEnb; ++i)
     {
         for (uint32_t j = 0; j < nUePerEnb; ++j)
         {
             uint32_t idx = i * nUePerEnb + j;
             lteHelper->Attach(ueLteDevs.Get(idx), enbLteDevs.Get(i));
         }
     }
 
     // Set default route for UEs
     for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
     {
         Ptr<Ipv4StaticRouting> ueStaticRouting =
             ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(u)->GetObject<Ipv4>());
         ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
     }
 
     // Bearers
     enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE; 
     EpsBearer bearer(q);
 
     //===========================
     // Create multiple flows
     //===========================
 
     // 1) VoIP-like flow (small pkts, high freq, e.g. 8kb/s ~ 24kb/s)
     // 2) Video-like flow (bigger pkts, moderate freq)
     // 3) Best-effort background (could be TCP or UDP)
 
     // remoteHost => UE: downlink traffic
     uint16_t dlPortBase = 4000; 
     ApplicationContainer apps; 
     double startTime = 0.1;
     double stopTime = simTime;
 
     // 依序給每個 UE 裝幾種 App
     for (uint32_t u = 0; u < ueNodes.GetN(); u++)
     {
         Ipv4Address ueAddr = ueIpIfaces.GetAddress(u);
 
         // (1) VoIP-like (UDP)
         {
             uint16_t dlPort = dlPortBase++;
             UdpClientHelper voipClient(ueAddr, dlPort);
             voipClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
             voipClient.SetAttribute("PacketSize", UintegerValue(160));  // small size
             voipClient.SetAttribute("Interval", TimeValue(MilliSeconds(20))); // 50 pkts/s => 8kbps
             
             ApplicationContainer voipServer;
             PacketSinkHelper voipSink("ns3::UdpSocketFactory",
                                       InetSocketAddress(Ipv4Address::GetAny(), dlPort));
             voipServer = voipSink.Install(ueNodes.Get(u));
             voipServer.Start(Seconds(startTime));
             voipServer.Stop(Seconds(stopTime));
             
             // client on remote host
             ApplicationContainer voipClientApp = voipClient.Install(remoteHost);
             voipClientApp.Start(Seconds(startTime));
             voipClientApp.Stop(Seconds(stopTime));
             apps.Add(voipServer);
             apps.Add(voipClientApp);
         }
 
         // (2) Video-like (UDP)
         {
             uint16_t dlPort = dlPortBase++;
             UdpClientHelper videoClient(ueAddr, dlPort);
             videoClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
             // e.g. 1200 bytes every 5ms => 1.92Mbps
             videoClient.SetAttribute("PacketSize", UintegerValue(1200));
             videoClient.SetAttribute("Interval", TimeValue(MilliSeconds(5)));
             
             ApplicationContainer videoServer;
             PacketSinkHelper videoSink("ns3::UdpSocketFactory",
                                        InetSocketAddress(Ipv4Address::GetAny(), dlPort));
             videoServer = videoSink.Install(ueNodes.Get(u));
             videoServer.Start(Seconds(startTime));
             videoServer.Stop(Seconds(stopTime));
 
             ApplicationContainer videoClientApp = videoClient.Install(remoteHost);
             videoClientApp.Start(Seconds(startTime));
             videoClientApp.Stop(Seconds(stopTime));
             apps.Add(videoServer);
             apps.Add(videoClientApp);
         }
 
         // (3) Best-effort (TCP)
         {
             uint16_t dlPort = dlPortBase++;
             BulkSendHelper bulkClient("ns3::TcpSocketFactory",
                                       InetSocketAddress(ueAddr, dlPort));
             bulkClient.SetAttribute("MaxBytes", UintegerValue(0)); // unlimited
 
             ApplicationContainer bulkServer;
             PacketSinkHelper bulkSink("ns3::TcpSocketFactory",
                                       InetSocketAddress(Ipv4Address::GetAny(), dlPort));
             bulkServer = bulkSink.Install(ueNodes.Get(u));
             bulkServer.Start(Seconds(startTime));
             bulkServer.Stop(Seconds(stopTime));
 
             ApplicationContainer bulkClientApp = bulkClient.Install(remoteHost);
             bulkClientApp.Start(Seconds(startTime));
             bulkClientApp.Stop(Seconds(stopTime));
             apps.Add(bulkServer);
             apps.Add(bulkClientApp);
         }
     }
 
     // Enable traces
     lteHelper->EnableTraces();
 
     // Optionally install FlowMonitor
     Ptr<FlowMonitor> monitor;
     FlowMonitorHelper flowmon;
     if (enableFlowMonitor)
     {
         monitor = flowmon.InstallAll();
     }
 
     // Stop
     Simulator::Stop(Seconds(simTime));
     Simulator::Run();
 
     //--- Write flow monitor results to CSV ---
     if (enableFlowMonitor)
     {
         monitor->CheckForLostPackets();
 
         Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
         map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();
 
         std::ofstream outFile;
         outFile.open(outputCsv.c_str());
         outFile << "FlowID,SrcAddr,DstAddr,TxPackets,RxPackets,Throughput(Kbps),AverageDelay(s)\n";
 
         for (auto &flow: stats)
         {
             FlowId id = flow.first;
             FlowMonitor::FlowStats st = flow.second;
             Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(id);
 
             double throughputKbps = 0.0;
             if (st.timeLastRxPacket.GetSeconds() - st.timeFirstTxPacket.GetSeconds() > 0)
             {
                 throughputKbps = (st.rxBytes * 8.0) /
                                   (st.timeLastRxPacket.GetSeconds() - st.timeFirstTxPacket.GetSeconds()) / 1024.0;
             }
 
             double avgDelay = 0.0;
             if (st.rxPackets > 0)
             {
                 avgDelay = st.delaySum.GetSeconds() / st.rxPackets;
             }
 
             outFile << id << ","
                     << t.sourceAddress << ","
                     << t.destinationAddress << ","
                     << st.txPackets << ","
                     << st.rxPackets << ","
                     << throughputKbps << ","
                     << avgDelay << "\n";
         }
         outFile.close();
     }
 
     Simulator::Destroy();
 
     NS_LOG_UNCOND("Done. Simulation finished.");
     return 0;
 }
 