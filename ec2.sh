#!/usr/bin/env bash
# ec2.sh — manage the sam3d EC2 instance
#
# Usage:
#   ./ec2.sh start    Start instance, wait until ready, print IP
#   ./ec2.sh stop     Stop instance, confirm stopped
#   ./ec2.sh status   Show state, uptime, estimated session cost
#   ./ec2.sh cost     Show last 7 days actual AWS spend by service
#   ./ec2.sh ip       Print current public IP
#   ./ec2.sh ssh      SSH into the instance

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
INSTANCE_ID="i-077e0d2fd19cf3776"      # TODO: fill in your instance ID
HOURLY_RATE="3.50"                     # TODO: check your instance's on-demand rate
KEY_FILE="$HOME/dev/sam3d-keypair.pem"
SSH_USER="ubuntu"
REGION="ap-northeast-1"               # Sydney — adjust if different
# ─────────────────────────────────────────────────────────────────────────────

AWS="aws --region $REGION"

_instance_state() {
    $AWS ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query "Reservations[0].Instances[0].State.Name" --output text
}

_instance_ip() {
    $AWS ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query "Reservations[0].Instances[0].PublicIpAddress" --output text
}

_instance_launch_time() {
    $AWS ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query "Reservations[0].Instances[0].LaunchTime" --output text
}

_instance_type() {
    $AWS ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query "Reservations[0].Instances[0].InstanceType" --output text
}

cmd_start() {
    local state
    state=$(_instance_state)

    if [[ "$state" == "running" ]]; then
        echo "Instance is already running."
        echo "  IP: $(_instance_ip)"
        return
    fi

    echo "Starting instance $INSTANCE_ID..."
    $AWS ec2 start-instances --instance-ids "$INSTANCE_ID" > /dev/null

    echo -n "Waiting for running state"
    while true; do
        state=$(_instance_state)
        if [[ "$state" == "running" ]]; then
            break
        fi
        echo -n "."
        sleep 3
    done
    echo " ready."

    echo -n "Waiting for SSH to be available"
    local ip
    ip=$(_instance_ip)
    while true; do
        if ssh -i "$KEY_FILE" -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
               "$SSH_USER@$ip" true 2>/dev/null; then
            break
        fi
        echo -n "."
        sleep 3
    done
    echo " up."

    echo ""
    echo "  Instance : $INSTANCE_ID ($(_instance_type))"
    echo "  IP       : $ip"
    echo "  SSH      : ssh -i $KEY_FILE $SSH_USER@$ip"
    echo "  API URL  : http://$ip:8000"
}

cmd_stop() {
    local state
    state=$(_instance_state)

    if [[ "$state" == "stopped" ]]; then
        echo "Instance is already stopped."
        return
    fi

    echo "Stopping instance $INSTANCE_ID..."
    $AWS ec2 stop-instances --instance-ids "$INSTANCE_ID" > /dev/null

    echo -n "Waiting for stopped state"
    while true; do
        state=$(_instance_state)
        if [[ "$state" == "stopped" ]]; then
            break
        fi
        echo -n "."
        sleep 3
    done
    echo " done."
    echo "  Instance stopped. Compute charges have ended."
}

cmd_status() {
    local state ip launch_time instance_type
    state=$(_instance_state)
    ip=$(_instance_ip)
    instance_type=$(_instance_type)

    echo "  Instance : $INSTANCE_ID ($instance_type)"
    echo "  State    : $state"
    echo "  IP       : ${ip:-none}"

    if [[ "$state" == "running" ]]; then
        launch_time=$(_instance_launch_time)

        # Calculate uptime in seconds (macOS-compatible)
        local launch_epoch now_epoch elapsed_secs elapsed_hrs cost
        launch_epoch=$(date -jf "%Y-%m-%dT%H:%M:%S" "${launch_time%%.*}" +%s 2>/dev/null \
                       || date -d "$launch_time" +%s)
        now_epoch=$(date +%s)
        elapsed_secs=$(( now_epoch - launch_epoch ))
        elapsed_hrs=$(echo "scale=2; $elapsed_secs / 3600" | bc)
        cost=$(echo "scale=4; $elapsed_hrs * $HOURLY_RATE" | bc)

        local hours mins
        hours=$(( elapsed_secs / 3600 ))
        mins=$(( (elapsed_secs % 3600) / 60 ))

        echo "  Started  : $launch_time"
        echo "  Uptime   : ${hours}h ${mins}m"
        echo "  Est. cost: \$$cost (@ \$$HOURLY_RATE/hr)"
    fi
}

cmd_cost() {
    local start_date end_date
    end_date=$(date +%Y-%m-%d)
    start_date=$(date -v-7d +%Y-%m-%d 2>/dev/null || date -d "7 days ago" +%Y-%m-%d)

    echo "AWS spend last 7 days (by service):"
    echo ""
    aws ce get-cost-and-usage \
        --time-period "Start=$start_date,End=$end_date" \
        --granularity MONTHLY \
        --metrics UnblendedCost \
        --group-by Type=DIMENSION,Key=SERVICE \
        --query "sort_by(ResultsByTime[0].Groups, &Metrics.UnblendedCost.Amount)[].[Keys[0], Metrics.UnblendedCost.Amount]" \
        --output table
}

cmd_ip() {
    local state ip
    state=$(_instance_state)
    ip=$(_instance_ip)
    if [[ "$state" != "running" ]]; then
        echo "Instance is not running (state: $state)"
        exit 1
    fi
    echo "$ip"
}

cmd_ssh() {
    local ip
    ip=$(_instance_ip)
    if [[ -z "$ip" || "$ip" == "None" ]]; then
        echo "Instance has no public IP — is it running?"
        exit 1
    fi
    exec ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "$SSH_USER@$ip"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${1:-}" in
    start)  cmd_start  ;;
    stop)   cmd_stop   ;;
    status) cmd_status ;;
    cost)   cmd_cost   ;;
    ip)     cmd_ip     ;;
    ssh)    cmd_ssh    ;;
    *)
        echo "Usage: $(basename "$0") {start|stop|status|cost|ip|ssh}"
        exit 1
        ;;
esac
